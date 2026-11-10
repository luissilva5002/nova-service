"""Routing tests for action lanes inside LLMEngine, without a model."""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_DATA_DIR = tempfile.TemporaryDirectory(prefix="nova-llm-action-lanes-", ignore_cleanup_errors=True)
os.environ.setdefault("NOVA_DATA_DIR", _DATA_DIR.name)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.brain import llm_engine as llm_engine_module  # noqa: E402
from nova.brain.generation import GenerationParams  # noqa: E402
from nova.brain.llm_engine import ACTION_LANE_INTENTS, LLMEngine  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


MUSIC = [tool("ytm_play"), tool("ytm_pause")]
CALENDAR = [tool("gc_list_today"), tool("gc_find")]


class FakeLaneEngine:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.action_calls: list[dict] = []
        self.warm_calls: list[tuple[str, list]] = []
        self.borrow_count = 0

    def action_turn(self, intent, tools, user_text, context_block, params, should_stop=None):
        self.action_calls.append(
            {"intent": intent, "tools": tools, "user_text": user_text,
             "context_block": context_block, "params": params}
        )
        if self.fail:
            raise RuntimeError("lane exploded")
        return types.SimpleNamespace(
            text='<tool_call>\n{"name": "ytm_play", "arguments": {"query": "jazz"}}\n</tool_call>',
            metrics={"action_lane": f"action:{intent}", "warm_prefill_tokens": 0},
        )

    def warm_action(self, intent, tools):
        self.warm_calls.append((intent, tools))
        return {"lane": intent, "source": "built", "tokens": 100, "prefilled_tokens": 100, "seconds": 0.1}

    @contextlib.contextmanager
    def borrow(self):
        self.borrow_count += 1
        yield


class FakeLlm:
    def __init__(self):
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "legacy reply"}}], "usage": {}}


class ActionLaneRoutingTests(unittest.TestCase):
    def make_engine(self, lane_engine=None):
        engine = LLMEngine()
        engine._loaded = True
        engine._llm = FakeLlm()
        engine._lanes = lane_engine if lane_engine is not None else FakeLaneEngine()
        engine._GenerationParams = GenerationParams
        return engine

    def generate(self, engine, intent="action_music", tools=MUSIC, history=None, user_text="play some jazz"):
        return run(
            engine.generate_with_tools(
                user_text,
                tools,
                memory_context="",
                history=history or [],
                intent=intent,
                session_id="session-1",
            )
        )

    def test_eligible_action_turn_goes_through_the_action_lane(self):
        engine = self.make_engine()
        decision = self.generate(engine)

        self.assertEqual(len(engine._lanes.action_calls), 1)
        call = engine._lanes.action_calls[0]
        self.assertEqual(call["intent"], "action_music")
        self.assertEqual(call["tools"], MUSIC)
        self.assertEqual(call["user_text"], "play some jazz")
        self.assertEqual(engine._llm.calls, [])  # legacy path untouched
        self.assertEqual(decision["type"], "text")
        self.assertIn("<tool_call>", decision["content"])
        self.assertEqual(decision["metrics"]["action_lane"], "action:action_music")
        self.assertNotIn("lane", decision["metrics"])

    def test_tool_temperature_and_token_budget_are_used(self):
        engine = self.make_engine()
        self.generate(engine)
        params = engine._lanes.action_calls[0]["params"]
        self.assertEqual(params.temperature, llm_engine_module.LLM_TEMP_TOOLS)
        self.assertEqual(params.max_tokens, llm_engine_module.LLM_MAX_TOKENS_TOOL)

    def test_single_tool_intent_stays_on_the_legacy_path(self):
        engine = self.make_engine()
        decision = self.generate(engine, intent="memory_recall", tools=[tool("recall_note")])
        self.assertEqual(engine._lanes.action_calls, [])
        self.assertEqual(len(engine._llm.calls), 1)
        self.assertEqual(decision["content"], "legacy reply")

    def test_chat_intent_and_empty_tools_are_not_action_lane_turns(self):
        engine = self.make_engine()
        self.assertFalse(engine._action_lane_eligible("chat", MUSIC))
        self.assertFalse(engine._action_lane_eligible("action_music", []))
        self.assertFalse(engine._action_lane_eligible("action_music", None))
        self.assertTrue(engine._action_lane_eligible("action_calendar", CALENDAR))

    def test_flag_off_uses_the_legacy_path(self):
        engine = self.make_engine()
        with patch.object(llm_engine_module, "LLM_KV_ACTION_LANES", False):
            self.generate(engine)
        self.assertEqual(engine._lanes.action_calls, [])
        self.assertEqual(len(engine._llm.calls), 1)

    def test_lanes_disabled_uses_the_legacy_path(self):
        engine = self.make_engine()
        engine._lanes = None
        decision = self.generate(engine)
        self.assertEqual(decision["content"], "legacy reply")

    def test_action_lane_failure_falls_back_to_legacy(self):
        engine = self.make_engine(FakeLaneEngine(fail=True))
        decision = self.generate(engine)
        self.assertEqual(len(engine._lanes.action_calls), 1)
        self.assertEqual(len(engine._llm.calls), 1)
        self.assertEqual(decision["content"], "legacy reply")
        self.assertEqual(engine._lanes.borrow_count, 1)  # legacy call still parks the resident lane

    def test_calendar_context_has_lookup_and_recent_exchange(self):
        engine = self.make_engine()
        history = [
            {"role": "user", "content": "what is on tomorrow"},
            {"role": "assistant", "content": "A dentist visit at 10."},
        ]
        with patch.object(llm_engine_module, "LLM_KV_ACTION_HISTORY_TURNS", 1):
            self.generate(engine, intent="action_calendar", tools=CALENDAR, history=history,
                          user_text="move it to 11")
        block = engine._lanes.action_calls[0]["context_block"]
        self.assertTrue(block.startswith("[Context]\nCurrent date:"))
        self.assertIn("Upcoming dates", block)
        self.assertIn("Recent conversation:\nUser: what is on tomorrow\nAssistant: A dentist visit at 10.", block)
        self.assertTrue(block.endswith("[/Context]"))

    def test_history_turns_zero_omits_recent_conversation(self):
        engine = self.make_engine()
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        with patch.object(llm_engine_module, "LLM_KV_ACTION_HISTORY_TURNS", 0):
            self.generate(engine, history=history)
        self.assertNotIn("Recent conversation", engine._lanes.action_calls[0]["context_block"])

    def test_surrogates_never_reach_the_lane_or_the_legacy_prompt(self):
        engine = self.make_engine()
        self.generate(engine, user_text="play Hora padr\udce3o")
        engine._lanes.action_calls[0]["user_text"].encode("utf-8")

        legacy = self.make_engine()
        self.generate(legacy, intent="memory_recall", tools=[tool("recall_note")],
                      history=[{"role": "user", "content": "x\udce3"}], user_text="y\udce3")
        for message in legacy._llm.calls[0]["messages"]:
            message["content"].encode("utf-8")

    def test_prewarm_builds_only_eligible_action_lanes(self):
        engine = self.make_engine()
        reports = run(
            engine.prewarm_action_lanes(
                {
                    "action_calendar": CALENDAR,
                    "action_music": MUSIC,
                    "memory_write": [tool("update_user_preference")],  # single tool: skipped
                    "memory_recall": [tool("recall_note")],            # not an action-lane intent
                }
            )
        )
        self.assertEqual([call[0] for call in engine._lanes.warm_calls], ["action_calendar", "action_music"])
        self.assertEqual(len(reports), 2)

    def test_prewarm_survives_a_failing_lane_and_does_nothing_without_lanes(self):
        class Flaky(FakeLaneEngine):
            def warm_action(self, intent, tools):
                if intent == "action_calendar":
                    raise RuntimeError("boom")
                return super().warm_action(intent, tools)

        engine = self.make_engine(Flaky())
        reports = run(engine.prewarm_action_lanes({"action_calendar": CALENDAR, "action_music": MUSIC}))
        self.assertEqual([report["lane"] for report in reports], ["action_music"])

        engine._lanes = None
        self.assertEqual(run(engine.prewarm_action_lanes({"action_music": MUSIC})), [])

    def test_prewarm_waits_while_a_user_turn_holds_the_lock_and_until_it_goes_quiet(self):
        engine = self.make_engine()
        built: list[float] = []
        real_warm = engine._lanes.warm_action

        def warm(intent, tools):
            built.append(time.monotonic())
            return real_warm(intent, tools)

        engine._lanes.warm_action = warm

        async def scenario():
            with patch.object(llm_engine_module, "PREWARM_IDLE_SECONDS", 0.3):
                async with engine._lock:  # a user turn is running
                    task = asyncio.create_task(
                        engine.prewarm_action_lanes({"action_calendar": CALENDAR})
                    )
                    await asyncio.sleep(0.2)
                    during_user_turn = list(built)
                finished = time.monotonic()
                await task
                return during_user_turn, finished

        during, finished = run(scenario())
        self.assertEqual(during, [])  # nothing was built while the user held the lock
        self.assertEqual(len(built), 1)
        self.assertGreaterEqual(built[0] - finished, 0.25)  # it waited out the quiet period

    def test_prewarm_starts_immediately_when_nothing_else_is_happening(self):
        engine = self.make_engine()
        started = time.monotonic()
        with patch.object(llm_engine_module, "PREWARM_IDLE_SECONDS", 30.0):
            reports = run(engine.prewarm_action_lanes({"action_music": MUSIC}))
        self.assertEqual(len(reports), 1)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_activity_lock_marks_user_work_but_not_background_work(self):
        async def scenario():
            lock = llm_engine_module._ActivityLock()
            self.assertEqual(lock.last_activity, float("-inf"))
            async with lock.background():
                self.assertTrue(lock.locked())
            self.assertEqual(lock.last_activity, float("-inf"))
            async with lock:
                entered = lock.last_activity
            return entered, lock.last_activity

        entered, exited = run(scenario())
        self.assertGreater(entered, float("-inf"))
        self.assertGreaterEqual(exited, entered)

    def test_action_lane_intents_are_the_multi_tool_intents(self):
        self.assertEqual(set(ACTION_LANE_INTENTS), {"action_calendar", "action_music", "memory_write"})


if __name__ == "__main__":
    unittest.main()
