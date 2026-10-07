"""Real-model tests for per-session synchronous chat-lane orchestration."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NOVA_LLM_MODEL", "qwen2.5-3b")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.config import LLM_MODELS  # noqa: E402
from nova.brain.context_manager import ContextManager, LlamaBackend  # noqa: E402
from nova.brain.generation import GenerationParams, TokenGenerator  # noqa: E402
from nova.brain.lane_engine import ChatTurnResult, LaneEngine, LaneOverflow  # noqa: E402

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
SYSTEM = "You are Nova, a concise assistant. Answer clearly and briefly."
PARAMS = GenerationParams(max_tokens=12, temperature=0.0)


class EvalCounter:
    """Count llama eval tokens until the first nonempty generated text delta."""

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

    def __exit__(self, _exc_type, _exc_value, _traceback):
        del self.llama.eval

    def stop_counting(self) -> None:
        self.active = False


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class LaneEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_threads=4,
            verbose=False,
        )
        cls.backend = LlamaBackend(cls.llama)

    @classmethod
    def tearDownClass(cls):
        cls.llama.close()

    def setUp(self):
        self.llama.reset()
        self.engine = self.make_engine()

    def tearDown(self):
        print(f"LaneEngine stats: {self.engine.stats()}")

    def make_engine(
        self,
        gen_reserve: int = 160,
        load_fraction: float = 0.5,
        compact_fraction: float = 0.35,
    ) -> LaneEngine:
        return LaneEngine(
            self.llama,
            n_ctx=1024,
            system_prompt=SYSTEM,
            gen_reserve=gen_reserve,
            load_fraction=load_fraction,
            compact_fraction=compact_fraction,
        )

    def run_stream(
        self,
        engine: LaneEngine,
        session_id: str,
        user_text: str,
        context_block: str | None = None,
        history_messages: list[dict] | None = None,
        should_stop=None,
    ) -> tuple[ChatTurnResult, int, str]:
        result = None
        deltas: list[str] = []
        stream = engine.chat_stream(
            session_id,
            user_text,
            context_block,
            history_messages,
            PARAMS,
            should_stop,
        )
        counter = EvalCounter(self.llama)
        try:
            with counter:
                for item in stream:
                    if isinstance(item, str):
                        deltas.append(item)
                        if item:
                            counter.stop_counting()
                    else:
                        result = item
        finally:
            stream.close()
        if result is None:
            raise AssertionError("chat_stream did not return a ChatTurnResult.")
        return result, counter.total, "".join(deltas)

    def run_chat(self, *args, **kwargs) -> ChatTurnResult:
        return self.engine.chat(*args, params=PARAMS, **kwargs)

    def completion_text(self, messages: list[dict]) -> str:
        self.llama.reset()
        response = self.llama.create_chat_completion(
            messages=messages,
            max_tokens=12,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()

    def test_t1_context_blocks_reuse_lane_and_commit_exact_ids(self):
        engine = self.make_engine()
        context = "[Context]\n" + ("Current date and useful reference details. " * 20) + "[/Context]"
        history: list[dict] = []
        initial_history_ids = None
        evaluations = []
        for turn in range(1, 7):
            before = (
                engine._lanes["session"].context.history_ids
                if engine.has_lane("session")
                else None
            )
            result, evaluated, _ = self.run_stream(
                engine,
                "session",
                f"Turn {turn}: give one short useful suggestion.",
                context,
                history,
            )
            self.assertTrue(result.committed)
            lane = engine._lanes["session"]
            if before is None:
                initial_history_ids = list(lane.context.history_ids)
            else:
                self.assertGreater(len(lane.context.history_ids), len(before))
                self.assertEqual(lane.context.history_ids[: len(before)], before)
            history.extend(
                (
                    {"role": "user", "content": f"Turn {turn}: give one short useful suggestion."},
                    {"role": "assistant", "content": result.text},
                )
            )
            evaluations.append(evaluated)
            self.assertEqual(engine.stats()["cold_starts"], 1)
        self.assertIsNotNone(initial_history_ids)
        self.assertTrue(all(value < 220 for value in evaluations[1:]))
        self.assertLessEqual(evaluations[5], evaluations[1] + 40)
        print(f"LaneEngine stats: {engine.stats()}")

    def test_t2_context_turn_matches_chat_completion(self):
        transcript = [
            {"role": "user", "content": "I enjoy short walks."},
            {"role": "assistant", "content": "Enjoy your short walks!"},
            {"role": "user", "content": "I prefer quiet parks."},
            {"role": "assistant", "content": "Relax in a quiet park."},
        ]
        context = "[Context]\nCurrent date: Monday, 2026-10-05\n[/Context]"
        current_user = "Suggest one thing I could do today."
        engine = self.make_engine()
        actual = engine.chat(
            "parity",
            current_user,
            context,
            transcript,
            PARAMS,
        )
        expected = self.completion_text(
            [
                {"role": "system", "content": SYSTEM},
                *transcript,
                {"role": "user", "content": f"{context}\n\n{current_user}"},
            ]
        )
        self.assertEqual(actual.text.strip(), expected)
        print(f"LaneEngine stats: {engine.stats()}")

    def test_t3_external_turn_is_included_in_following_prompt(self):
        first = self.run_chat("external", "I like quiet gardens.", None, [])
        transcript = [
            {"role": "user", "content": "I like quiet gardens."},
            {"role": "assistant", "content": first.text},
        ]
        self.engine.record_external_turn(
            "external",
            "I also enjoy birdwatching.",
            "That is a peaceful hobby.",
        )
        transcript.extend(
            (
                {"role": "user", "content": "I also enjoy birdwatching."},
                {"role": "assistant", "content": "That is a peaceful hobby."},
            )
        )
        current_user = "Suggest a relaxing weekend activity."
        actual = self.run_chat("external", current_user, None, transcript)
        expected = self.completion_text(
            [
                {"role": "system", "content": SYSTEM},
                *transcript,
                {"role": "user", "content": current_user},
            ]
        )
        self.assertEqual(actual.text.strip(), expected)
        self.assertEqual(self.engine.stats()["external_turns"], 1)

    def test_t4_restart_rebuilds_history_then_reuses_it(self):
        messages: list[dict] = []
        original_results = []
        for user_text in ("I am planning a park visit.", "I like shaded paths."):
            result = self.run_chat("original", user_text, None, messages)
            original_results.append(result)
            messages.extend(
                (
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": result.text},
                )
            )

        restarted = self.make_engine()
        first_user = "What should I bring?"
        first, first_evaluated, _ = self.run_stream(
            restarted,
            "restored",
            first_user,
            history_messages=messages,
        )
        second_user = "What is a good time to go?"
        second, second_evaluated, _ = self.run_stream(
            restarted,
            "restored",
            second_user,
            history_messages=messages,
        )
        self.assertGreaterEqual(first_evaluated, first.metrics["prompt_tokens"] - 1)
        self.assertLess(second_evaluated, 220)

        control = self.make_engine()
        for user_text in ("I am planning a park visit.", "I like shaded paths."):
            control.chat("control", user_text, None, [], PARAMS)
        expected_first = control.chat("control", first_user, None, messages, PARAMS)
        expected_second = control.chat("control", second_user, None, messages, PARAMS)
        self.assertEqual(first.text, expected_first.text)
        self.assertEqual(second.text, expected_second.text)
        self.assertEqual(len(original_results), 2)

    def test_t5_streaming_and_cancellation(self):
        deltas_result, _, deltas = self.run_stream(
            self.engine,
            "stream",
            "Explain briefly why leaves change color.",
        )
        self.assertEqual(deltas, deltas_result.text)
        self.assertTrue(deltas_result.committed)

        lane = self.engine._lanes["stream"]
        before = lane.context.history_ids
        checks = 0

        def should_stop():
            nonlocal checks
            checks += 1
            return checks >= 3

        cancelled, _, _ = self.run_stream(
            self.engine,
            "stream",
            "Continue with three more details.",
            should_stop=should_stop,
        )
        self.assertEqual(cancelled.finish_reason, "cancelled")
        self.assertFalse(cancelled.committed)
        self.assertEqual(lane.context.history_ids, before)

    def test_t6_compaction_rebuilds_from_supplied_history(self):
        engine = self.make_engine(gen_reserve=64)
        history: list[dict] = []
        for turn in range(24):
            user_text = f"Turn {turn}: " + ("describe a peaceful garden in detail. " * 12)
            result = engine.chat(
                "compact",
                user_text,
                None,
                history,
                GenerationParams(max_tokens=12, temperature=0.0),
            )
            self.assertTrue(result.committed)
            history.extend(
                (
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": result.text},
                )
            )
        self.assertGreater(engine.stats()["compactions"], 0)
        self.assertTrue(engine.has_lane("compact"))
        self.assertTrue(engine._lanes["compact"].context.history_ids)
        print(f"LaneEngine stats: {engine.stats()}")

    def test_t7_overflow_leaves_engine_ready_for_next_turn(self):
        with self.assertRaises(LaneOverflow):
            self.run_chat("overflow", "A long request. " * 3000, None, [])
        self.assertGreaterEqual(self.engine.stats()["overflows"], 1)
        self.assertTrue(self.engine.has_lane("overflow"))
        recovered = self.run_chat("overflow", "Reply with one short greeting.", None, [])
        self.assertTrue(recovered.committed)

    def test_t8_borrow_preserves_chat_lane_for_next_turn(self):
        first = self.run_chat("borrow", "I am planning a quiet walk.", None, [])
        helper = ContextManager(self.backend, n_ctx=1024, gen_reserve=64)
        helper.start_session("You are an unrelated helper.")
        helper_prompt = helper.begin_turn("Give one short fact about the moon.")
        with self.engine.borrow():
            helper_result = TokenGenerator(
                self.llama,
                {helper.end_of_turn_id, int(self.llama.token_eos())},
            ).generate(
                helper_prompt,
                GenerationParams(max_tokens=12, temperature=0.0),
            )
        self.assertTrue(helper_result.ids)
        next_user = "What should I bring on the walk?"
        actual, evaluated, _ = self.run_stream(
            self.engine,
            "borrow",
            next_user,
        )

        control = self.make_engine()
        control_first = control.chat("control", "I am planning a quiet walk.", None, [], PARAMS)
        expected = control.chat("control", next_user, None, [], PARAMS)
        self.assertLess(evaluated, 220)
        self.assertEqual(first.text, control_first.text)
        self.assertEqual(actual.text, expected.text)

    def test_t9_metrics_include_expected_fields(self):
        result = self.run_chat("metrics", "Say hello briefly.", None, [])
        expected = {
            "completion_tokens",
            "prompt_tokens",
            "elapsed_seconds",
            "total_elapsed_seconds",
            "tokens_per_second",
            "decode_tokens_per_second",
            "ttft_seconds",
            "cache_hit_tokens",
            "evaluated_tokens",
            "lane",
        }
        self.assertTrue(expected.issubset(result.metrics))
        stop_ids = {self.engine._lanes["metrics"].context.end_of_turn_id, int(self.llama.token_eos())}
        non_stop_ids = [token_id for token_id in result.ids if token_id not in stop_ids]
        self.assertEqual(result.metrics["completion_tokens"], len(non_stop_ids))
        self.assertLessEqual(result.metrics["ttft_seconds"], result.metrics["elapsed_seconds"])
        self.assertEqual(result.metrics["lane"], "chat:4:chat:7:metrics")


if __name__ == "__main__":
    unittest.main(verbosity=2)
