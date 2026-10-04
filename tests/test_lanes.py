"""Real-model tests for lane-local llama.cpp KV residency."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("NOVA_LLM_MODEL", "qwen2.5-3b")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.config import LLM_MODELS  # noqa: E402
from nova.brain.context_manager import ContextManager, LlamaBackend  # noqa: E402
from nova.brain.generation import (  # noqa: E402
    GenerationParams,
    GenerationResult,
    TokenGenerator,
    stop_ids_for,
)
from nova.brain import lanes as lanes_module  # noqa: E402
from nova.brain.lanes import LaneManager, LaneSpec  # noqa: E402

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
MAX_TOKENS = 12
CHAT_SPEC = LaneSpec(
    name="chat",
    system_prompt="You are a concise assistant. Answer clearly and briefly.",
    gen_reserve=64,
)
ACTION_SPEC = LaneSpec(
    name="music",
    system_prompt="You control a music player. Answer briefly.",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "ytm_play",
                "description": "Play a song or genre.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ],
    gen_reserve=64,
)


class EvalCounter:
    """Count eval tokens until the first nonempty streamed text delta."""

    def __init__(self, llama) -> None:
        self.llama = llama
        self.total = 0
        self.active = True
        self._original_eval = None

    def __enter__(self):
        self._original_eval = self.llama.eval

        def counted(tokens, *args, **kwargs):
            if self.active:
                self.total += len(tokens)
            return self._original_eval(tokens, *args, **kwargs)

        self.llama.eval = counted
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.llama.eval

    def stop_counting(self) -> None:
        self.active = False


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class LaneManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_threads=4,
            verbose=False,
        )
        cls.backend = LlamaBackend(cls.llama)
        cls.stop_ids = stop_ids_for(cls._empty_context(cls.backend), cls.llama)
        cls.generator = TokenGenerator(cls.llama, cls.stop_ids)

    @classmethod
    def tearDownClass(cls):
        cls.llama.close()

    @staticmethod
    def _empty_context(backend):
        context = ContextManager(backend, n_ctx=1024, gen_reserve=64)
        context.start_session("End token helper.")
        return context

    def setUp(self):
        self.llama.reset()

    def make_manager(self, max_snapshot_bytes=256 * 1024 * 1024):
        return LaneManager(
            self.llama,
            self.backend,
            n_ctx=1024,
            max_snapshot_bytes=max_snapshot_bytes,
        )

    def generate_turn(self, manager, lane, user_text, count_eval=True):
        with manager.use(lane):
            prompt = lane.context.begin_turn(user_text)
            result = None
            counter = EvalCounter(self.llama) if count_eval else None
            stream = self.generator.stream(
                prompt,
                GenerationParams(max_tokens=MAX_TOKENS, temperature=0.0),
            )
            try:
                if counter is None:
                    items = stream
                    for item in items:
                        if isinstance(item, GenerationResult):
                            result = item
                else:
                    with counter:
                        for item in stream:
                            if isinstance(item, str) and item:
                                counter.stop_counting()
                            elif isinstance(item, GenerationResult):
                                result = item
                if result is None:
                    raise AssertionError("TokenGenerator did not return GenerationResult.")
            finally:
                stream.close()
            lane.context.commit_turn(result.ids)
            return result, counter.total if counter is not None else result.evaluated_tokens

    def generate_action(self, manager, lane, user_text):
        with manager.use(lane):
            prompt = lane.context.begin_turn(user_text)
            try:
                result, evaluated = self._consume_current_turn(prompt)
            finally:
                lane.context.abort_turn()
        return result, evaluated

    def _consume_current_turn(self, prompt):
        result = None
        counter = EvalCounter(self.llama)
        stream = self.generator.stream(
            prompt,
            GenerationParams(max_tokens=MAX_TOKENS, temperature=0.0),
        )
        try:
            with counter:
                for item in stream:
                    if isinstance(item, str) and item:
                        counter.stop_counting()
                    elif isinstance(item, GenerationResult):
                        result = item
        finally:
            stream.close()
        if result is None:
            raise AssertionError("TokenGenerator did not return GenerationResult.")
        return result, counter.total

    def print_stats(self, manager):
        print(f"LaneManager stats: {manager.stats()}")

    def test_t1_interleaved_chat_action_lanes_keep_chat_prefix(self):
        manager = self.make_manager()
        chat_a = manager.chat_lane("A", CHAT_SPEC)
        chat_b = manager.chat_lane("B", CHAT_SPEC)
        music = manager.action_lane(ACTION_SPEC)

        self.generate_turn(manager, chat_a, "I am planning a quiet walk in the park.")
        self.generate_turn(manager, chat_b, "Tell me a short fact about the moon.")
        self.generate_action(manager, music, "Play some jazz.")
        actual, evaluated = self.generate_turn(
            manager,
            chat_a,
            "What should I bring on the walk?",
        )

        self.llama.reset()
        control = LaneManager(self.llama, self.backend, n_ctx=1024)
        control_a = control.chat_lane("control", CHAT_SPEC)
        self.generate_turn(control, control_a, "I am planning a quiet walk in the park.")
        expected, _ = self.generate_turn(
            control,
            control_a,
            "What should I bring on the walk?",
        )

        self.assertLess(evaluated, 60)
        self.assertEqual(actual.ids, expected.ids)
        self.print_stats(manager)

    def test_t2_warmed_action_reuses_base_prefix_across_calls_and_chat(self):
        manager = self.make_manager()
        music = manager.action_lane(ACTION_SPEC)
        system_tokens = len(music.context.system_ids)
        self.assertGreater(manager.warm(music), 0)
        self.assertEqual(manager.warm(music), 0)

        for text in ("Play some jazz.", "Play quiet piano.", "Play upbeat music."):
            _, evaluated = self.generate_action(manager, music, text)
            self.assertLess(evaluated, 60)
            self.assertLess(evaluated, system_tokens)

        chat = manager.chat_lane("between-actions", CHAT_SPEC)
        self.generate_turn(manager, chat, "What is a short nature fact?")
        _, evaluated = self.generate_action(manager, music, "Play acoustic guitar.")
        self.assertLess(evaluated, 60)
        self.assertLess(evaluated, system_tokens)
        self.print_stats(manager)

    def test_t3_borrow_does_not_lose_resident_chat_lane(self):
        manager = self.make_manager()
        chat = manager.chat_lane("A", CHAT_SPEC)
        self.generate_turn(manager, chat, "I am planning a quiet walk in the park.")

        helper = ContextManager(self.backend, n_ctx=1024, gen_reserve=64)
        helper.start_session("You are an unrelated helper.")
        helper_prompt = helper.begin_turn(
            "This unrelated request contains enough words to create a roughly "
            "one-hundred-and-fifty-token prompt. " * 20
        )
        self.assertGreaterEqual(len(helper_prompt), 150)
        with manager.borrow():
            helper_result = self.generator.generate(
                helper_prompt,
                GenerationParams(max_tokens=MAX_TOKENS, temperature=0.0),
            )
        self.assertTrue(helper_result.ids)

        actual, evaluated = self.generate_turn(
            manager,
            chat,
            "What should I bring on the walk?",
        )
        self.llama.reset()
        control = self.make_manager()
        control_chat = control.chat_lane("control", CHAT_SPEC)
        self.generate_turn(control, control_chat, "I am planning a quiet walk in the park.")
        expected, _ = self.generate_turn(
            control,
            control_chat,
            "What should I bring on the walk?",
        )
        self.assertLess(evaluated, 60)
        self.assertEqual(actual.ids, expected.ids)
        self.print_stats(manager)

    def test_t4_evicted_lane_cold_starts_and_remains_correct(self):
        manager = self.make_manager(max_snapshot_bytes=1)
        lane_a = manager.chat_lane("A", CHAT_SPEC)
        lane_b = manager.chat_lane("B", CHAT_SPEC)

        first, _ = self.generate_turn(manager, lane_a, "I am planning a quiet walk.")
        self.generate_turn(manager, lane_b, "Tell me one short fact about the moon.")
        stats = manager.stats()
        self.assertGreater(stats["evictions"], 0)
        self.assertFalse(lane_a.warm)

        returned, evaluated = self.generate_turn(
            manager,
            lane_a,
            "What should I bring?",
        )
        self.assertGreaterEqual(evaluated, returned.prompt_tokens - 1)

        self.llama.reset()
        control = self.make_manager()
        control_a = control.chat_lane("control", CHAT_SPEC)
        self.generate_turn(control, control_a, "I am planning a quiet walk.")
        expected, _ = self.generate_turn(control, control_a, "What should I bring?")
        self.assertEqual(returned.ids, expected.ids)
        self.assertTrue(first.ids)
        self.print_stats(manager)

    def test_t5_loaded_history_cold_prefills_then_reuses(self):
        history = [
            {"role": "user", "content": "I am planning a quiet walk."},
            {"role": "assistant", "content": "That sounds relaxing."},
            {"role": "user", "content": "I prefer tree-lined routes."},
            {"role": "assistant", "content": "A shaded trail would suit you."},
            {"role": "user", "content": "I would like lunch afterward."},
            {"role": "assistant", "content": "You could find a cafe nearby."},
        ]
        manager = self.make_manager()
        lane = manager.chat_lane("restored", CHAT_SPEC, history=history)
        first, evaluated_first = self.generate_turn(
            manager,
            lane,
            "Suggest a simple plan.",
        )
        self.assertEqual(evaluated_first, first.prompt_tokens)

        second, evaluated_second = self.generate_turn(
            manager,
            lane,
            "What should I bring?",
        )
        self.assertLess(evaluated_second, 60)
        self.assertTrue(second.ids)
        self.print_stats(manager)

    def test_t6_restore_failure_falls_back_to_cold_and_remains_correct(self):
        manager = self.make_manager()
        lane_a = manager.chat_lane("A", CHAT_SPEC)
        lane_b = manager.chat_lane("B", CHAT_SPEC)
        self.generate_turn(manager, lane_a, "I am planning a quiet walk.")
        self.generate_turn(manager, lane_b, "Tell me one short fact about the moon.")

        original_restore = lanes_module.restore_snapshot
        calls = 0

        def fail_once(llama, snapshot):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated restore failure")
            return original_restore(llama, snapshot)

        with patch.object(lanes_module, "restore_snapshot", side_effect=fail_once):
            actual, _ = self.generate_turn(manager, lane_a, "What should I bring?")

        self.assertEqual(manager.stats()["restore_failures"], 1)
        self.assertFalse(lane_a.warm)
        self.llama.reset()
        control = self.make_manager()
        control_a = control.chat_lane("control", CHAT_SPEC)
        self.generate_turn(control, control_a, "I am planning a quiet walk.")
        expected, _ = self.generate_turn(control, control_a, "What should I bring?")
        self.assertEqual(actual.ids, expected.ids)
        self.print_stats(manager)

    def test_t7_nested_use_spec_conflict_and_forget(self):
        manager = self.make_manager()
        lane_a = manager.chat_lane("A", CHAT_SPEC)
        lane_b = manager.chat_lane("B", CHAT_SPEC)
        self.generate_turn(manager, lane_a, "I am planning a quiet walk.")
        self.generate_turn(manager, lane_b, "Tell me one short fact about the moon.")
        self.assertTrue(lane_a.warm)

        with manager.use(lane_a):
            with self.assertRaisesRegex(RuntimeError, "nested lane use"):
                with manager.use(lane_b):
                    pass

        manager.action_lane(ACTION_SPEC)
        with self.assertRaises(ValueError):
            manager.action_lane(
                LaneSpec(
                    name=ACTION_SPEC.name,
                    system_prompt=ACTION_SPEC.system_prompt,
                    tools=[],
                    gen_reserve=ACTION_SPEC.gen_reserve,
                )
            )

        snapshots_before = manager.stats()["snapshots"]
        manager.forget_chat("A")
        self.assertEqual(manager.stats()["snapshots"], snapshots_before - 1)
        self.assertFalse(lane_a.warm)
        self.assertIsNot(manager.chat_lane("A", CHAT_SPEC), lane_a)
        self.print_stats(manager)

    def test_t8_repeated_resident_use_is_a_stats_no_op(self):
        manager = self.make_manager()
        lane = manager.chat_lane("A", CHAT_SPEC)
        self.generate_turn(manager, lane, "I am planning a quiet walk.")
        before = manager.stats()
        for _ in range(3):
            with manager.use(lane):
                pass
            self.assertEqual(manager.stats(), before)
        self.print_stats(manager)


if __name__ == "__main__":
    unittest.main(verbosity=2)
