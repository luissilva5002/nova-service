"""Lane-state wiring tests for nova.main without loading an inference model."""
from __future__ import annotations

import asyncio
import importlib.machinery
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

_DATA_DIR = tempfile.TemporaryDirectory(prefix="nova-main-lanes-")
os.environ["NOVA_DATA_DIR"] = _DATA_DIR.name


def _module(name: str, *, package: bool = False, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name,
        loader=None,
        is_package=package,
    )
    if package:
        module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    sys.modules[name] = module
    return module


STUBBED_MODULES: list[str] = []
try:
    import fastapi  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            pass

        def mount(self, *args, **kwargs):
            pass

        def _route(self, *args, **kwargs):
            return lambda function: function

        on_event = _route
        get = _route
        post = _route
        websocket = _route

    class _HTTPException(Exception):
        pass

    class _WebSocketDisconnect(Exception):
        pass

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            pass

    _module(
        "fastapi",
        package=True,
        FastAPI=_FastAPI,
        HTTPException=_HTTPException,
        WebSocket=_Placeholder,
        WebSocketDisconnect=_WebSocketDisconnect,
    )
    _module("fastapi.middleware", package=True)
    _module("fastapi.middleware.cors", CORSMiddleware=_Placeholder)
    _module("fastapi.staticfiles", StaticFiles=_Placeholder)
    _module("fastapi.responses", HTMLResponse=_Placeholder)
    STUBBED_MODULES.extend(
        (
            "fastapi",
            "fastapi.middleware",
            "fastapi.middleware.cors",
            "fastapi.staticfiles",
            "fastapi.responses",
        )
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova import main  # noqa: E402
from nova.memory import session_memory  # noqa: E402


class FakeEngine:
    def __init__(self, lanes_enabled: bool, decision: dict | None = None):
        self.lanes_enabled = lanes_enabled
        self._loaded = True
        self.decision = decision or {
            "type": "text",
            "content": "Test reply.",
            "metrics": {},
        }
        self.generate_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.external_calls: list[tuple[str, str, str]] = []
        self.forget_calls: list[str] = []
        self._lanes_present: set[str] = set()

    async def generate_with_tools(self, **kwargs):
        self.generate_calls.append(kwargs)
        self._lanes_present.add(kwargs["session_id"])
        return self.decision

    async def stream_chat(self, *args, **kwargs):
        self.stream_calls.append({"args": args, **kwargs})
        yield {"type": "text_delta", "text": "A streamed reply."}
        yield {
            "type": "done",
            "text": "A streamed reply.",
            "metrics": {"completion_tokens": 3},
        }

    async def record_external_turn(self, session_id, user_text, assistant_text):
        self.external_calls.append((session_id, user_text, assistant_text))

    async def forget_lane(self, session_id):
        self.forget_calls.append(session_id)
        self._lanes_present.discard(session_id)

    def has_lane(self, session_id):
        return session_id in self._lanes_present

    def lane_stats(self):
        return {"enabled": self.lanes_enabled}

    def list_models(self):
        return []


class FakeTTS:
    _loaded = True

    async def synthesize_stream(self, text):
        yield b""

    def sample_rate(self):
        return 16000

    def list_available_voices(self):
        return []


class FakeRouter:
    tool_schemas = [
        {
            "type": "function",
            "function": {"name": "ytm_play", "parameters": {"type": "object"}},
        }
    ]

    @staticmethod
    def dispatch(name, arguments):
        return "Playing jazz."


def run(coro):
    return asyncio.run(coro)


class MainLaneTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.engine = FakeEngine(lanes_enabled=True)
        self.histories: dict[str, list[dict]] = {}
        self.context_limits: list[int] = []
        self.engine.decision = {
            "type": "text",
            "content": "Test reply.",
            "metrics": {},
        }
        self.stack.enter_context(patch.object(main, "llm_engine", self.engine))
        self.stack.enter_context(patch.object(main, "tts_engine", FakeTTS()))
        self.stack.enter_context(patch.object(main, "core_store", MagicMock()))
        self.stack.enter_context(patch.object(main, "tool_router", FakeRouter()))
        self.stack.enter_context(
            patch.object(
                main,
                "classify_intent",
                new=self._classify_intent,
            )
        )
        self.stack.enter_context(
            patch.object(
                main,
                "filter_tools_for_intent",
                side_effect=lambda schemas, intent: schemas if intent.startswith("action_") else [],
            )
        )
        self.stack.enter_context(patch.object(main, "wants_vault_context", return_value=False))
        self.stack.enter_context(
            patch.object(main.vault_store, "get_profile_fields", return_value={})
        )
        self.stack.enter_context(
            patch.object(main.vault_store, "get_context_snippet", return_value={
                "preferences": {},
                "relevant_knowledge": [],
                "knowledge_context": "",
            })
        )
        self.stack.enter_context(
            patch.object(main, "_answer_from_memory", return_value=None)
        )
        self.stack.enter_context(
            patch.object(main, "get_session_summary", return_value="test summary")
        )
        self.append_turn_patcher = patch.object(
            main, "append_turn", side_effect=self._append_turn
        )
        self.get_context_patcher = patch.object(
            main, "get_context", side_effect=self._get_context
        )
        self.append_turn_patcher.start()
        self.get_context_patcher.start()
        self.addCleanup(self.append_turn_patcher.stop)
        self.addCleanup(self.get_context_patcher.stop)
        self.stack.enter_context(patch.object(main, "clear_session"))
        self.stack.enter_context(patch.object(main.memory_agent, "process_turn", return_value=None))

    async def _classify_intent(self, user_text, llm):
        return {
            "intent": getattr(self, "intent", "chat"),
            "source": "test",
            "confidence": 1.0,
        }

    def _append_turn(self, session_id, role, content):
        self.histories.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )

    def _get_context(self, session_id, limit=20):
        self.context_limits.append(limit)
        return list(self.histories.get(session_id, []))[-limit:]

    def test_t1_lane_chat_uses_full_persisted_history_and_stores_pair(self):
        session_id = "seeded-lane-history"
        memory_dir = Path(_DATA_DIR.name) / "session-memory-t1"
        memory_dir.mkdir(parents=True, exist_ok=True)
        previous_memory_dir = session_memory.SESSION_MEMORY_DIR
        session_memory.SESSION_MEMORY_DIR = memory_dir
        self.addCleanup(
            setattr,
            session_memory,
            "SESSION_MEMORY_DIR",
            previous_memory_dir,
        )
        self.append_turn_patcher.stop()
        self.get_context_patcher.stop()
        session_memory.clear_session(session_id)
        for index in range(15):
            session_memory.append_turn(session_id, "user", f"Seed user {index}.")
            session_memory.append_turn(session_id, "assistant", f"Seed reply {index}.")

        self.engine.decision = {
            "type": "text",
            "content": "Lane answer.",
            "metrics": {"lane": "chat:test"},
        }
        result = run(main.run_pipeline("What next?", session_id))
        self.assertEqual(result[0], "Lane answer.")
        call = self.engine.generate_calls[-1]
        self.assertEqual(call["session_id"], session_id)
        self.assertEqual(len(call["history"]), 30)
        self.assertEqual(call["history"][0]["content"], "Seed user 0.")
        stored = json.loads(
            session_memory._session_file(session_id).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [(message["role"], message["content"]) for message in stored["history"][-2:]],
            [("user", "What next?"), ("assistant", "Lane answer.")],
        )
        self.assertEqual(self.engine.external_calls, [])

    def test_t2_legacy_chat_decision_is_recorded_once(self):
        run(main.run_pipeline("Tell me something.", "legacy"))
        self.assertEqual(
            self.engine.external_calls,
            [("legacy", "Tell me something.", "Test reply.")],
        )

    def test_t3_action_tool_result_is_recorded(self):
        self.intent = "action_music"
        self.engine.decision = {
            "type": "tool_call",
            "name": "ytm_play",
            "arguments": {"query": "jazz"},
            "metrics": {},
        }
        result = run(main.run_pipeline("Play jazz.", "action"))
        self.assertEqual(result[0], "Playing jazz.")
        self.assertEqual(
            self.engine.external_calls,
            [("action", "Play jazz.", "Playing jazz.")],
        )

    def test_t4_session_commands_sync_or_forget_lane(self):
        run(main.run_pipeline("show summary", "summary-session"))
        self.assertEqual(len(self.engine.external_calls), 1)
        self.assertEqual(self.engine.external_calls[0][:2], ("summary-session", "show summary"))
        self.assertTrue(self.engine.external_calls[0][2])

        external_count = len(self.engine.external_calls)
        run(main.run_pipeline("clear the chat history", "clear-session"))
        self.assertEqual(self.engine.forget_calls, ["clear-session"])
        self.assertEqual(len(self.engine.external_calls), external_count)

    def test_t5_memory_recall_direct_answer_syncs_lane(self):
        self.intent = "memory_recall"
        with patch.object(main, "_answer_from_memory", return_value="I recall it."):
            result = run(main.run_pipeline("What do you remember?", "memory"))
        self.assertEqual(result[0], "I recall it.")
        self.assertEqual(
            self.engine.external_calls,
            [("memory", "What do you remember?", "I recall it.")],
        )

    def test_t6_disabled_lanes_keep_12_message_history_without_sync(self):
        self.engine.lanes_enabled = False
        self.histories["off"] = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(30)
        ]
        run(main.run_pipeline("A normal question.", "off"))
        self.assertEqual(len(self.engine.generate_calls[-1]["history"]), 12)
        self.assertEqual(self.engine.generate_calls[-1]["session_id"], "off")
        self.assertEqual(self.engine.external_calls, [])
        self.assertEqual(self.engine.forget_calls, [])

    def test_t7_websocket_stream_receives_session_and_long_history(self):
        session_id = "ws-session"
        self.histories[session_id] = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(30)
        ]
        disconnect_type = main.WebSocketDisconnect

        class FakeWebSocket:
            def __init__(self, current_session_id):
                self.query_params = {"session_id": current_session_id}
                self._nova_session_id = current_session_id
                self.sent = []
                self.received = False

            async def accept(self):
                pass

            async def receive_text(self):
                if not self.received:
                    self.received = True
                    return json.dumps(
                        {"type": "text", "text": "hello", "stream": True}
                    )
                raise disconnect_type()

            async def send_text(self, text):
                self.sent.append(json.loads(text))

        websocket = FakeWebSocket(session_id)
        run(main.websocket_endpoint(websocket))
        call = self.engine.stream_calls[-1]
        self.assertEqual(call["session_id"], session_id)
        self.assertEqual(len(call["history"]), 30)
        self.assertIn(400, self.context_limits)
        self.assertTrue(any(message["type"] == "text" for message in websocket.sent))
        self.assertTrue(any(message["type"] == "metrics" for message in websocket.sent))

        self.engine.lanes_enabled = False
        self.histories["ws-off-session"] = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(30)
        ]
        off_websocket = FakeWebSocket("ws-off-session")
        run(main.websocket_endpoint(off_websocket))
        self.assertEqual(len(self.engine.stream_calls[-1]["history"]), 12)
        self.assertEqual(self.engine.stream_calls[-1]["session_id"], "ws-off-session")
        self.assertEqual(self.context_limits[-1], 12)

    def test_t8_status_reports_lane_stats(self):
        response = run(main.get_status())
        self.assertEqual(response["kv_lanes"], {"enabled": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
