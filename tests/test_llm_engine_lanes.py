"""Integration tests for the opt-in LLMEngine KV-lane path."""
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_DATA_DIR = tempfile.TemporaryDirectory(prefix="nova-llm-lanes-")
os.environ["NOVA_LLM_MODEL"] = "qwen2.5-3b"
os.environ["NOVA_LLM_CTX"] = "2048"
os.environ["NOVA_DATA_DIR"] = _DATA_DIR.name

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.config import LLM_MODELS  # noqa: E402
from nova.brain import llm_engine as llm_engine_module  # noqa: E402
from nova.brain.llm_engine import LLMEngine  # noqa: E402

MODEL_PATH = LLM_MODELS["qwen2.5-3b"]["path"]
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = Llama is not None
except (ImportError, OSError):
    LLAMA_AVAILABLE = False

SKIP_REASON = (
    "Requires llama-cpp-python and the configured Qwen2.5-3B GGUF "
    f"(llama_cpp available={LLAMA_AVAILABLE}, model exists={MODEL_PATH.is_file()})."
)


def run(coro):
    return asyncio.run(coro)


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class LLMEngineLanes01OffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flag_patcher = patch.object(llm_engine_module, "LLM_KV_LANES", False)
        cls.flag_patcher.start()
        cls.engine = LLMEngine()
        cls.engine.load()

    @classmethod
    def tearDownClass(cls):
        if cls.engine._llm is not None:
            cls.engine._llm.close()
        cls.flag_patcher.stop()

    def test_t1_flag_off_preserves_legacy_generation_and_helpers(self):
        self.assertFalse(self.engine.lanes_enabled)
        self.assertEqual(self.engine.lane_stats(), {"enabled": False})
        response = run(
            self.engine.generate_with_tools(
                "Reply with a short greeting.",
                [],
                history=[],
                intent="chat",
                session_id="off-session",
            )
        )
        self.assertEqual(response["type"], "text")
        self.assertTrue(response["content"])
        run(
            self.engine.record_external_turn(
                "off-session",
                "External question.",
                "External response.",
            )
        )
        self.assertFalse(self.engine.has_lane("off-session"))


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class LLMEngineLanes02EnabledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flag_patcher = patch.object(llm_engine_module, "LLM_KV_LANES", True)
        cls.flag_patcher.start()
        cls.engine = LLMEngine()
        cls.engine.load()

    @classmethod
    def tearDownClass(cls):
        if cls.engine._llm is not None:
            cls.engine._llm.close()
        cls.flag_patcher.stop()

    def setUp(self):
        self.engine._llm.reset()

    @property
    def llama(self):
        return self.engine._llm

    def chat(self, session_id: str, user_text: str, history=None, memory_context=""):
        return run(
            self.engine.generate_with_tools(
                user_text,
                [],
                memory_context=memory_context,
                history=history or [],
                intent="chat",
                session_id=session_id,
            )
        )

    def test_t2_flag_on_drops_llama_ram_cache(self):
        self.assertTrue(self.engine.lanes_enabled)
        self.assertIsNone(self.llama.cache)

    def test_t3_six_turns_reuse_history_with_memory_context(self):
        session_id = "reuse-session"
        history: list[dict] = []
        memory = (
            '{"preferences":["quiet walks","gardens","birdwatching"],'
            '"home":"near the river","schedule":"usually free in the morning",'
            '"interests":["nature","reading","local history"]}'
        )
        metrics = []
        for turn in range(1, 7):
            response = self.chat(
                session_id,
                f"Turn {turn}: give one concise suggestion based on what you know.",
                history,
                memory,
            )
            self.assertEqual(response["type"], "text")
            self.assertTrue(response["content"])
            metrics.append(response["metrics"])
            history.extend(
                (
                    {
                        "role": "user",
                        "content": f"Turn {turn}: give one concise suggestion based on what you know.",
                    },
                    {"role": "assistant", "content": response["content"]},
                )
            )
        self.assertTrue(
            all(
                {"completion_tokens", "prompt_tokens", "elapsed_seconds",
                 "total_elapsed_seconds", "tokens_per_second", "cache_hit_tokens"}
                <= metric.keys()
                for metric in metrics
            )
        )
        self.assertTrue(all(metric["evaluated_tokens"] < 250 for metric in metrics[1:]))
        self.assertLessEqual(
            metrics[5]["evaluated_tokens"],
            metrics[1]["evaluated_tokens"] + 40,
        )

    def test_t4_context_block_matches_legacy_message_format(self):
        session_id = "parity-session"
        user_text = "Suggest something to do today."
        memory = '{"hobbies":["walking","gardening"],"preference":"quiet places"}'
        captured = {}
        original_chat = self.engine._lanes.chat

        def capture(*args, **kwargs):
            captured["session_id"] = args[0]
            captured["user_text"] = args[1]
            captured["context_block"] = args[2]
            return original_chat(*args, **kwargs)

        history = [
            {"role": "user", "content": "I like being outside."},
            {"role": "assistant", "content": "A walk may be enjoyable."},
        ]
        with (
            patch.object(
                LLMEngine,
                "_current_date_context",
                return_value="Current date: Monday, 2026-10-05",
            ),
            patch.object(self.engine._lanes, "chat", side_effect=capture),
        ):
            response = self.chat(session_id, user_text, history, memory)
            legacy_last = self.engine._build_messages(
                user_text,
                memory,
                [],
                history,
                "chat",
            )[-1]["content"]
        self.assertEqual(captured["session_id"], session_id)
        self.assertEqual(captured["user_text"], user_text)
        self.assertEqual(
            captured["context_block"] + "\n\n" + captured["user_text"],
            legacy_last,
        )
        self.assertTrue(response["content"])

    def test_t5_raw_helper_preserves_chat_lane(self):
        session_id = "helper-session"
        history: list[dict] = []
        first = self.chat(session_id, "I enjoy quiet walks.", history)
        history.extend(
            (
                {"role": "user", "content": "I enjoy quiet walks."},
                {"role": "assistant", "content": first["content"]},
            )
        )
        second = self.chat(session_id, "I prefer tree-lined routes.", history)
        history.extend(
            (
                {"role": "user", "content": "I prefer tree-lined routes."},
                {"role": "assistant", "content": second["content"]},
            )
        )
        run(
            self.engine.generate_raw(
                "You are an unrelated helper.",
                "Give one short fact about the moon.",
                max_tokens=12,
            )
        )
        third = self.chat(session_id, "What should I bring?", history)
        self.assertLess(third["metrics"]["evaluated_tokens"], 250)

    def test_t6_external_turn_does_not_start_a_new_lane(self):
        session_id = "external-session"
        history: list[dict] = []
        first = self.chat(session_id, "I enjoy birdwatching.", history)
        history.extend(
            (
                {"role": "user", "content": "I enjoy birdwatching."},
                {"role": "assistant", "content": first["content"]},
            )
        )
        cold_starts = self.engine.lane_stats()["cold_starts"]
        run(
            self.engine.record_external_turn(
                session_id,
                "I also like quiet gardens.",
                "Gardens can be peaceful places to watch birds.",
            )
        )
        history.extend(
            (
                {"role": "user", "content": "I also like quiet gardens."},
                {"role": "assistant", "content": "Gardens can be peaceful places to watch birds."},
            )
        )
        result = self.chat(session_id, "Suggest a relaxing activity.", history)
        self.assertTrue(result["content"])
        self.assertEqual(self.engine.lane_stats()["cold_starts"], cold_starts)

    def test_t7_lane_failure_falls_back_and_next_call_works(self):
        session_id = "fallback-session"
        with patch.object(
            self.engine._lanes,
            "chat",
            side_effect=RuntimeError("simulated lane failure"),
        ):
            response = self.chat(session_id, "Say hello briefly.")
        self.assertEqual(response["type"], "text")
        self.assertTrue(response["content"])
        self.assertFalse(self.engine.has_lane(session_id))
        next_response = self.chat(session_id, "Give one short greeting.")
        self.assertEqual(next_response["type"], "text")
        self.assertTrue(next_response["content"])

    def test_t8_stream_reuses_lane_and_cancels_on_early_close(self):
        session_id = "stream-session"

        async def collect(user_text):
            return [
                event
                async for event in self.engine.stream_chat(
                    user_text,
                    history=[],
                    intent="chat",
                    session_id=session_id,
                )
            ]

        first_events = run(collect("I like quiet parks."))
        self.assertTrue(first_events[-1]["type"] == "done")
        self.assertTrue(first_events[-1]["metrics"])
        self.assertTrue(all(event["type"] == "text_delta" for event in first_events[:-1]))

        second_events = run(collect("What should I bring on a walk?"))
        self.assertEqual(second_events[-1]["type"], "done")
        self.assertLess(second_events[-1]["metrics"]["evaluated_tokens"], 250)

        lane = self.engine._lanes._lanes[session_id]
        history_before = lane.context.history_ids

        async def consume_two_then_close():
            stream = self.engine.stream_chat(
                "Explain how trees grow.",
                history=[],
                intent="chat",
                session_id=session_id,
            )
            deltas = 0
            try:
                async for event in stream:
                    if event["type"] == "text_delta":
                        deltas += 1
                    if deltas == 2:
                        return
            finally:
                await stream.aclose()

        run(consume_two_then_close())
        self.assertEqual(lane.context.history_ids, history_before)
        next_result = self.chat(session_id, "Say one short sentence about trees.")
        self.assertTrue(next_result["content"])
        self.assertTrue(self.engine.has_lane(session_id))

    def test_t9_qwen3_guard_leaves_lanes_disabled(self):
        original_model_id = self.engine._active_model_id
        self.engine._active_model_id = "qwen3-1.7b"
        try:
            self.engine._init_lanes()
            self.assertIsNone(self.engine._lanes)
            self.assertFalse(self.engine.lanes_enabled)
        finally:
            self.engine._active_model_id = original_model_id
            self.engine._init_lanes()


if __name__ == "__main__":
    unittest.main(verbosity=2)
