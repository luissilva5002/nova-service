"""main.py: tool-call recovery, sanitizing and websocket error handling."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_main_lanes as harness  # noqa: E402  (stubs fastapi and imports nova.main)

main = harness.main
FakeEngine = harness.FakeEngine
FakeTTS = harness.FakeTTS


def run(coro):
    return asyncio.run(coro)


def tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


class RecordingRouter:
    def __init__(self, names):
        self.tool_schemas = [tool(name) for name in names]
        self.dispatched: list[tuple[str, dict]] = []

    def dispatch(self, name, arguments):
        self.dispatched.append((name, arguments))
        return "Done."


class FakeWebSocket:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent: list[dict] = []
        self.query_params = {"session_id": "ws-test"}
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self._frames:
            raise main.WebSocketDisconnect()
        frame = self._frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame

    async def send_text(self, text):
        self.sent.append(json.loads(text))


class ToolCallRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.engine = FakeEngine(lanes_enabled=True)
        self.router = RecordingRouter(["ytm_play", "ytm_pause"])
        self.histories: dict[str, list[dict]] = {}
        enter = self.stack.enter_context
        enter(patch.object(main, "llm_engine", self.engine))
        enter(patch.object(main, "tts_engine", FakeTTS()))
        enter(patch.object(main, "core_store", MagicMock()))
        enter(patch.object(main, "tool_router", self.router))
        enter(patch.object(main, "classify_intent", new=self._classify))
        enter(patch.object(main, "filter_tools_for_intent", side_effect=lambda schemas, intent: schemas))
        enter(patch.object(main, "wants_vault_context", return_value=False))
        enter(patch.object(main.vault_store, "get_profile_fields", return_value={}))
        enter(patch.object(main, "_answer_from_memory", return_value=None))
        enter(patch.object(main, "get_session_summary", return_value=""))
        enter(patch.object(main, "append_turn", side_effect=self._append))
        enter(patch.object(main, "get_context", side_effect=lambda sid, limit=20: []))
        enter(patch.object(main, "clear_session"))
        enter(patch.object(main.memory_agent, "process_turn", return_value=None))

    async def _classify(self, text, engine):
        return {"intent": "action_music", "source": "test", "confidence": 1.0}

    def _append(self, session_id, role, content):
        self.histories.setdefault(session_id, []).append({"role": role, "content": content})

    def decide(self, content):
        self.engine.decision = {"type": "text", "content": content, "metrics": {}}
        return run(main.run_pipeline("play some jazz", "tool-session"))

    def test_observed_unbalanced_call_is_recovered_and_dispatched(self):
        text = '<tool_call>\n{{"name": "ytm_play", "arguments": {"query": "jazz"}}\n</tool_call>'
        reply, *_ = self.decide(text)
        self.assertEqual(self.router.dispatched, [("ytm_play", {"query": "jazz"})])
        self.assertEqual(reply, "Done.")

    def test_unparseable_call_never_reaches_the_user_or_tts(self):
        reply, *_ = self.decide("<tool_call>\n{{garbage")
        self.assertEqual(reply, main._TOOL_PARSE_FAILURE_REPLY)
        self.assertNotIn("tool_call", reply)
        self.assertEqual(self.router.dispatched, [])
        stored = self.histories["tool-session"][-1]["content"]
        self.assertEqual(stored, main._TOOL_PARSE_FAILURE_REPLY)

    def test_call_to_a_tool_that_was_not_exposed_is_rejected(self):
        text = '<tool_call>{"name": "gc_delete_event", "arguments": {}}</tool_call>'
        reply, *_ = self.decide(text)
        self.assertEqual(reply, main._TOOL_PARSE_FAILURE_REPLY)
        self.assertEqual(self.router.dispatched, [])

    def test_plain_text_replies_are_untouched(self):
        reply, *_ = self.decide("Sure, here you go.")
        self.assertEqual(reply, "Sure, here you go.")

    def test_lone_surrogates_are_removed_before_the_model_sees_the_text(self):
        self.engine.decision = {"type": "text", "content": "ok", "metrics": {}}
        run(main.run_pipeline("play Hora padr\udce3o", "surrogate-session"))
        sent = self.engine.generate_calls[0]["user_text"]
        sent.encode("utf-8")
        self.assertIn("\ufffd", sent)

    def test_extract_helper_accepts_allowed_names(self):
        text = '<tool_call>{"name": "ytm_pause", "arguments": {}}</tool_call>'
        self.assertEqual(main._extract_tool_call_from_text(text), {"name": "ytm_pause", "arguments": {}})
        self.assertIsNone(main._extract_tool_call_from_text(text, {"ytm_play"}))


class WebSocketErrorHandlingTests(unittest.TestCase):
    def test_a_failed_turn_sends_an_error_frame_and_keeps_the_socket_open(self):
        calls = {"count": 0}

        async def flaky_pipeline(user_text, session_id="default"):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("model exploded")
            return "All good.", b"", {"completion_tokens": 1}, None

        socket = FakeWebSocket([
            json.dumps({"type": "text", "text": "first"}),
            json.dumps({"type": "text", "text": "second"}),
        ])
        with patch.object(main, "run_pipeline", flaky_pipeline):
            run(main.websocket_endpoint(socket))

        types = [frame["type"] for frame in socket.sent]
        self.assertEqual(types, ["error", "text", "metrics"])
        self.assertEqual(socket.sent[0]["text"], main._TURN_ERROR_REPLY)
        self.assertEqual(socket.sent[1]["text"], "All good.")
        self.assertEqual(calls["count"], 2)

    def test_disconnect_during_error_reply_ends_the_loop_cleanly(self):
        async def failing_pipeline(user_text, session_id="default"):
            raise RuntimeError("boom")

        socket = FakeWebSocket([json.dumps({"type": "text", "text": "hello"})])

        async def failing_send(text):
            raise RuntimeError("socket closed")

        socket.send_text = failing_send
        with patch.object(main, "run_pipeline", failing_pipeline):
            run(main.websocket_endpoint(socket))  # must not raise

    def test_non_string_text_is_reported_instead_of_crashing(self):
        socket = FakeWebSocket([json.dumps({"type": "text", "text": None})])
        with patch.object(main, "run_pipeline", AsyncMock()):
            run(main.websocket_endpoint(socket))
        self.assertEqual([frame["type"] for frame in socket.sent], ["error"])


if __name__ == "__main__":
    unittest.main()
