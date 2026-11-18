"""Tests for KV-C-lite: removing the per-turn [Context] block from the live KV cache.

These tests need no model and no llama-cpp-python: they use a fake llama whose
KV cache is a position -> token dict that follows llama.cpp's seq_rm / seq_add
semantics, and a ChatML fake backend for ContextManager.

Run from the repository root:
    python tests/test_kv_strip.py -v
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nova.brain import lane_engine as lane_engine_module  # noqa: E402
from nova.brain.context_manager import ContextManager  # noqa: E402
from nova.brain.generation import GenerationParams  # noqa: E402
from nova.brain.kv_edit import (  # noqa: E402
    KvEditCorrupted,
    KvEditRefused,
    align_context_span,
    kv_remove_span,
    strip_context,
)
from nova.brain.lane_engine import LaneEngine  # noqa: E402

IM_START, IM_END, EOS = 1, 2, 99999
CONTEXT_BLOCK = "[Context]\nCurrent date: Friday, 2026-11-13 (UTC+00:00)\n\nKnown user context: likes jazz and tea\n[/Context]"


class FakeBackend:
    """ChatML renderer + word tokenizer with <|im_start|>/<|im_end|> as single tokens."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = 100 + len(self._vocab)
        return self._vocab[piece]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            if chunk == "<|im_start|>":
                ids.append(IM_START)
            elif chunk == "<|im_end|>":
                ids.append(IM_END)
            else:
                ids.extend(self._id(piece) for piece in re.findall(r"\n+| +|\w+|[^\w\s]", chunk))
        return ids

    def render(self, messages, tools, add_generation_prompt) -> str:
        out = ""
        for message in messages:
            content = message["content"]
            if message["role"] == "system" and tools:
                content += "\nTOOLS " + repr(tools)
            out += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


class FakeCtx:
    """KV cache as {position: token}, with llama.cpp seq_rm / seq_add semantics."""

    def __init__(self) -> None:
        self.kv: dict[int, int] = {}
        self.rm_result: bool | None = True
        self.shift_error: Exception | None = None
        self.calls: list[tuple] = []

    def kv_cache_seq_rm(self, seq_id, p0, p1):
        self.calls.append(("rm", seq_id, p0, p1))
        if self.rm_result is False:
            return False
        p0 = max(p0, 0)
        p1 = 10**9 if p1 < 0 else p1
        for position in [p for p in self.kv if p0 <= p < p1]:
            del self.kv[position]
        return True

    def kv_cache_seq_shift(self, seq_id, p0, p1, delta):
        self.calls.append(("shift", seq_id, p0, p1, delta))
        if self.shift_error is not None:
            raise self.shift_error
        moved = {p + delta: t for p, t in self.kv.items() if p0 <= p < p1}
        for position in [p for p in self.kv if p0 <= p < p1]:
            del self.kv[position]
        self.kv.update(moved)


class FakeLlama:
    def __init__(self, n_ctx: int = 1024) -> None:
        self._n_ctx = n_ctx
        self.input_ids = np.zeros(n_ctx, dtype=np.intc)
        self.n_tokens = 0
        self._ctx = FakeCtx()
        self.resets = 0
        self.eval_counts: list[int] = []

    def n_ctx(self) -> int:
        return self._n_ctx

    def token_eos(self) -> int:
        return EOS

    def detokenize(self, ids):
        return b"w "

    def reset(self) -> None:
        self.n_tokens = 0
        self._ctx.kv.clear()
        self.resets += 1

    def _eval(self, tokens) -> None:
        for token in tokens:
            self.input_ids[self.n_tokens] = token
            self._ctx.kv[self.n_tokens] = int(token)
            self.n_tokens += 1

    def generate(self, tokens, **_kwargs):
        tokens = [int(t) for t in tokens]
        shared = 0
        while shared < min(self.n_tokens, len(tokens)) and int(self.input_ids[shared]) == tokens[shared]:
            shared += 1
        if shared == len(tokens):
            shared -= 1
        self.n_tokens = shared
        self._ctx.kv_cache_seq_rm(-1, shared, -1)
        self._eval(tokens[shared:])
        self.eval_counts.append(len(tokens) - shared)
        for token in (200 + len(tokens) % 7, 300, 301, IM_END):
            yield token
            self._eval([token])  # runs only if the consumer keeps going, like llama.cpp

    def kv_consistent(self) -> bool:
        return self._ctx.kv == {i: int(self.input_ids[i]) for i in range(self.n_tokens)}


def make_llama_with_turn(context_block: str | None = CONTEXT_BLOCK, user_text: str = "tell me about tea"):
    """A fake llama whose KV holds history + shown user segment + a short reply."""
    backend = FakeBackend()
    context = ContextManager(backend, n_ctx=1024, gen_reserve=64)
    context.start_session("You are NOVA.", None)
    llama = FakeLlama()
    prompt = context.begin_turn(user_text, context_block, context_position="before")
    llama._eval(prompt + [200, 300, 301])
    context.commit_turn([200, 300, 301, IM_END])
    return llama, context


class AlignTests(unittest.TestCase):
    def test_pure_insertion(self):
        self.assertEqual(align_context_span([1, 5, 77, 88, 9, 2], [1, 5, 9, 2]), (2, 2))

    def test_no_context_returns_none(self):
        self.assertIsNone(align_context_span([1, 5, 9, 2], [1, 5, 9, 2]))

    def test_boundary_merge_returns_none(self):
        self.assertIsNone(align_context_span([1, 5, 66, 88, 2], [1, 5, 9, 2]))

    def test_repeated_tokens_stay_unambiguous(self):
        shown, clean = [1, 7, 7, 7, 7, 2], [1, 7, 7, 2]
        offset, removed = align_context_span(shown, clean)
        self.assertEqual(shown[:offset] + shown[offset + removed :], clean)


class RemoveSpanTests(unittest.TestCase):
    def _llama(self, n: int = 20) -> FakeLlama:
        llama = FakeLlama()
        llama._eval(list(range(1000, 1000 + n)))
        return llama

    def test_removes_middle_and_slides_tail(self):
        llama = self._llama()
        expected = list(range(1000, 1005)) + list(range(1008, 1020))
        kv_remove_span(llama, 5, 8)
        self.assertEqual(llama.n_tokens, 17)
        self.assertEqual([int(t) for t in llama.input_ids[:17]], expected)
        self.assertTrue(llama.kv_consistent())
        self.assertIn(("shift", 0, 8, 20, -3), llama._ctx.calls)

    def test_removing_the_tail_needs_no_shift(self):
        llama = self._llama()
        kv_remove_span(llama, 15, 20)
        self.assertEqual(llama.n_tokens, 15)
        self.assertTrue(llama.kv_consistent())
        self.assertFalse(any(call[0] == "shift" for call in llama._ctx.calls))

    def test_out_of_range_is_refused(self):
        llama = self._llama()
        with self.assertRaises(KvEditRefused):
            kv_remove_span(llama, 5, 25)
        self.assertEqual(llama.n_tokens, 20)

    def test_rm_false_leaves_everything_untouched(self):
        llama = self._llama()
        llama._ctx.rm_result = False
        with self.assertRaises(KvEditRefused):
            kv_remove_span(llama, 5, 8)
        self.assertEqual(llama.n_tokens, 20)
        self.assertTrue(llama.kv_consistent())

    def test_missing_shift_api_is_refused_before_any_change(self):
        llama = self._llama()
        FakeCtx.kv_cache_seq_shift, saved = None, FakeCtx.kv_cache_seq_shift
        try:
            with self.assertRaises(KvEditRefused):
                kv_remove_span(llama, 5, 8)
        finally:
            FakeCtx.kv_cache_seq_shift = saved
        self.assertEqual(llama.n_tokens, 20)
        self.assertTrue(llama.kv_consistent())

    def test_shift_failure_is_reported_as_corruption(self):
        llama = self._llama()
        llama._ctx.shift_error = RuntimeError("boom")
        with self.assertRaises(KvEditCorrupted):
            kv_remove_span(llama, 5, 8)


class StripContextTests(unittest.TestCase):
    def test_strip_restores_kv_to_history(self):
        llama, context = make_llama_with_turn()
        info = context.last_commit
        before = llama.n_tokens
        result = strip_context(llama, context.history_ids[: info.user_start], info.shown, info.clean)
        self.assertEqual(result.status, "stripped", result.reason)
        self.assertEqual(result.removed_tokens, len(info.shown) - len(info.clean))
        self.assertEqual(llama.n_tokens, before - result.removed_tokens)
        history = context.history_ids
        self.assertEqual([int(t) for t in llama.input_ids[: llama.n_tokens]], history[: llama.n_tokens])
        self.assertTrue(llama.kv_consistent())

    def test_turn_without_context_block_is_skipped(self):
        llama, context = make_llama_with_turn(context_block=None)
        info = context.last_commit
        result = strip_context(llama, context.history_ids[: info.user_start], info.shown, info.clean)
        self.assertEqual((result.status, result.reason), ("skipped", "no_context"))

    def test_kv_that_differs_from_history_is_not_touched(self):
        llama, context = make_llama_with_turn()
        info = context.last_commit
        llama.input_ids[2] += 1
        snapshot = (llama.n_tokens, dict(llama._ctx.kv))
        result = strip_context(llama, context.history_ids[: info.user_start], info.shown, info.clean)
        self.assertEqual((result.status, result.reason), ("skipped", "kv_differs_from_history"))
        self.assertEqual((llama.n_tokens, llama._ctx.kv), snapshot)

    def test_kv_shorter_than_segment_is_skipped(self):
        llama, context = make_llama_with_turn()
        info = context.last_commit
        llama.n_tokens = info.user_start + 1
        result = strip_context(llama, context.history_ids[: info.user_start], info.shown, info.clean)
        self.assertEqual((result.status, result.reason), ("skipped", "kv_shorter_than_segment"))

    def test_shift_failure_resets_the_context(self):
        llama, context = make_llama_with_turn()
        info = context.last_commit
        llama._ctx.shift_error = RuntimeError("boom")
        result = strip_context(llama, context.history_ids[: info.user_start], info.shown, info.clean)
        self.assertEqual(result.status, "failed")
        self.assertEqual(llama.n_tokens, 0)
        self.assertEqual(llama.resets, 1)


class CommitInfoTests(unittest.TestCase):
    def test_last_commit_describes_the_user_segment(self):
        backend = FakeBackend()
        context = ContextManager(backend, n_ctx=1024, gen_reserve=64)
        context.start_session("You are NOVA.", None)
        self.assertIsNone(context.last_commit)
        context.begin_turn("hello there", CONTEXT_BLOCK, context_position="before")
        self.assertIsNone(context.last_commit)
        context.commit_turn([200, IM_END])
        info = context.last_commit
        self.assertEqual(info.user_start, len(context.system_ids))
        self.assertGreater(len(info.shown), len(info.clean))
        history = context.history_ids
        self.assertEqual(history[info.user_start : info.user_start + len(info.clean)], info.clean)
        context.begin_turn("again", None)
        self.assertIsNone(context.last_commit)


class LaneEngineStripTests(unittest.TestCase):
    TURNS = ["tell me about tea", "and what about coffee", "which one has more caffeine", "thanks, that helps"]

    def setUp(self):
        self._original_backend = lane_engine_module.LlamaBackend
        lane_engine_module.LlamaBackend = lambda llama: FakeBackend()

    def tearDown(self):
        lane_engine_module.LlamaBackend = self._original_backend

    def _run(self, strip: bool):
        llama = FakeLlama()
        engine = LaneEngine(llama, 1024, "You are NOVA.", gen_reserve=64, strip_context=strip)
        rows = []
        for text in self.TURNS:
            result = engine.chat("s1", text, CONTEXT_BLOCK, None, GenerationParams(max_tokens=16, temperature=0.0))
            history = engine._lanes["s1"].context.history_ids
            matches = [int(t) for t in llama.input_ids[: llama.n_tokens]] == history[: llama.n_tokens]
            rows.append((result.metrics["evaluated_tokens"], result.metrics.get("kv_strip"), matches, llama.kv_consistent()))
        return engine, llama, rows

    def test_strip_reduces_re_evaluation_and_keeps_kv_equal_to_history(self):
        engine_on, llama_on, rows_on = self._run(True)
        _engine_off, _llama_off, rows_off = self._run(False)
        for index in range(1, len(self.TURNS)):
            self.assertLess(rows_on[index][0], rows_off[index][0], f"turn {index + 1}")
        for evaluated, status, matches, consistent in rows_on:
            self.assertEqual(status, "stripped")
            self.assertTrue(matches and consistent)
        self.assertFalse(any(row[2] for row in rows_off[:-1]), "without strip the KV diverges from history")
        stats = engine_on.stats()
        self.assertEqual(stats["strip_ok"], len(self.TURNS))
        self.assertGreater(stats["strip_tokens_removed"], 0)
        self.assertEqual(llama_on.resets, 1)  # only the cold start of the lane

    def test_turns_after_strip_evaluate_only_the_new_message(self):
        _engine, llama, rows = self._run(True)
        backend = FakeBackend()
        context = ContextManager(backend, n_ctx=1024, gen_reserve=64)
        context.start_session("You are NOVA.", None)
        new_segment = len(backend.encode(context._user_segment(f"{CONTEXT_BLOCK}\n\n{self.TURNS[1]}")))
        # Re-evaluated: the previous reply's end-of-turn tokens (the stop token is never
        # evaluated by llama.cpp), the new shown user segment and the generation prompt.
        # The rest of the previous exchange is reused.
        expected = len(context._suffix_ids) + new_segment + len(context._gen_ids)
        self.assertEqual(rows[1][0], expected)

    def test_flag_off_does_nothing(self):
        engine, llama, rows = self._run(False)
        self.assertTrue(all(row[1] is None for row in rows))
        self.assertEqual(engine.stats()["strip_ok"] + engine.stats()["strip_skipped"], 0)

    def test_failed_strip_does_not_fail_the_turn(self):
        llama = FakeLlama()
        engine = LaneEngine(llama, 1024, "You are NOVA.", gen_reserve=64, strip_context=True)
        llama._ctx.shift_error = RuntimeError("boom")
        first = engine.chat("s1", "tell me about tea", CONTEXT_BLOCK, None, GenerationParams(max_tokens=16))
        self.assertEqual(first.finish_reason, "stop")
        self.assertTrue(first.metrics["kv_strip"].startswith("failed"))
        llama._ctx.shift_error = None
        second = engine.chat("s1", "and coffee", CONTEXT_BLOCK, None, GenerationParams(max_tokens=16))
        self.assertEqual(second.finish_reason, "stop")
        self.assertEqual(engine.stats()["strip_failed"], 1)


if __name__ == "__main__":
    unittest.main()
