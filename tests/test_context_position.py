"""Tests for context placement and externally generated turn appends."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NOVA_LLM_MODEL", "qwen2.5-3b")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.brain.context_manager import ContextManager, ContextOverflow, LlamaBackend  # noqa: E402
from nova.config import LLM_MODELS  # noqa: E402

MODEL_PATH = LLM_MODELS["qwen2.5-3b"]["path"]
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except (ImportError, OSError):
    Llama = None
    LLAMA_AVAILABLE = False

SKIP_REASON = (
    "Requires llama-cpp-python and the configured Qwen2.5-3B GGUF "
    f"(llama_cpp available={LLAMA_AVAILABLE}, model exists={MODEL_PATH.is_file()})."
)
SYSTEM = "You are Nova, a concise personal assistant."
USER_TEXT = "What should I bring on the walk?"
ASSISTANT_TEXT = "Bring water and comfortable shoes."
CONTEXT_BLOCK = "[Context]\nCurrent date: Sunday, 2026-10-04\n[/Context]"


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class ContextPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama = Llama(model_path=str(MODEL_PATH), vocab_only=True, verbose=False)
        cls.backend = LlamaBackend(cls.llama)

    @classmethod
    def tearDownClass(cls):
        cls.llama.close()

    def make_context(self, n_ctx: int = 4096, gen_reserve: int = 256):
        context = ContextManager(self.backend, n_ctx=n_ctx, gen_reserve=gen_reserve)
        context.start_session(SYSTEM)
        return context

    def test_t1_before_layout_is_shown_but_not_kept_in_history(self):
        context = self.make_context()
        hist_before = len(context.history_ids)
        prompt = context.begin_turn(
            USER_TEXT,
            CONTEXT_BLOCK,
            context_position="before",
        )
        clean = self.backend.encode(context._user_segment(USER_TEXT))
        shown = self.backend.encode(
            context._user_segment(f"{CONTEXT_BLOCK}\n\n{USER_TEXT}")
        )
        self.assertEqual(prompt[-len(context._gen_ids) - len(shown) : -len(context._gen_ids)], shown)

        reply = self.backend.encode(ASSISTANT_TEXT)
        context.commit_turn(reply)
        self.assertEqual(
            context.history_ids,
            list(context.system_ids)
            + clean
            + context._gen_ids
            + context._close(reply),
        )

        next_prompt = context.begin_turn("And anything else?")
        self.assertEqual(
            common_prefix(next_prompt, prompt + reply),
            hist_before + common_prefix(clean, shown),
        )

    def test_t2_default_context_position_remains_after(self):
        implicit = self.make_context()
        explicit = self.make_context()
        self.assertEqual(
            implicit.begin_turn(USER_TEXT, CONTEXT_BLOCK),
            explicit.begin_turn(USER_TEXT, CONTEXT_BLOCK, context_position="after"),
        )

    def test_t3_invalid_context_position_leaves_turn_closed(self):
        context = self.make_context()
        with self.assertRaises(ValueError):
            context.begin_turn(USER_TEXT, CONTEXT_BLOCK, context_position="middle")
        self.assertIsNone(context._turn)
        context.begin_turn(USER_TEXT)

    def test_t4_external_turn_matches_live_commit(self):
        external = self.make_context()
        live = self.make_context()
        external.append_external_turn(USER_TEXT, ASSISTANT_TEXT)
        live.begin_turn(USER_TEXT)
        live.commit_turn(self.backend.encode(ASSISTANT_TEXT))
        self.assertEqual(external.history_ids, live.history_ids)

    def test_t5_external_turn_overflow_is_atomic(self):
        context = self.make_context(n_ctx=128, gen_reserve=64)
        before = context.history_ids
        with self.assertRaises(ContextOverflow):
            context.append_external_turn("A user message " * 100, ASSISTANT_TEXT)
        self.assertEqual(context.history_ids, before)
        self.assertIsNone(context._turn)

    def test_t6_external_and_live_turns_remain_append_only(self):
        context = self.make_context()
        context.begin_turn("First live turn.")
        context.commit_turn(self.backend.encode("First live reply."))
        first_history = context.history_ids

        context.append_external_turn("External user turn.", "External assistant reply.")
        self.assertEqual(context.history_ids[: len(first_history)], first_history)
        after_external = context.history_ids

        prompt = context.begin_turn("Final live turn.")
        self.assertEqual(prompt[: len(after_external)], after_external)
        context.commit_turn(self.backend.encode("Final live reply."))
        self.assertEqual(context.history_ids[: len(after_external)], after_external)

    def test_t7_empty_external_text_is_rejected(self):
        context = self.make_context()
        for user_text, assistant_text in (
            ("", ASSISTANT_TEXT),
            ("   ", ASSISTANT_TEXT),
            (USER_TEXT, ""),
            (USER_TEXT, "\t "),
        ):
            with self.subTest(user_text=user_text, assistant_text=assistant_text):
                with self.assertRaises(ValueError):
                    context.append_external_turn(user_text, assistant_text)
        self.assertEqual(context.history_ids, context.system_ids)
        self.assertIsNone(context._turn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
