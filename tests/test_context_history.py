"""Tests for rebuilding ContextManager history from persisted chat text."""
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


def user_text(index: int) -> str:
    return f"Message {index}: " + "we are planning a relaxed Saturday outing with a walk and lunch. " * (
        1 + index % 3
    )


def reply_text(index: int) -> str:
    return f"Reply {index}: sounds good, a walk first and lunch after."


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class ContextHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama = Llama(model_path=str(MODEL_PATH), vocab_only=True, verbose=False)
        cls.backend = LlamaBackend(cls.llama)

    @classmethod
    def tearDownClass(cls):
        cls.llama.close()

    def make_context(self, n_ctx: int = 4096, gen_reserve: int = 256) -> ContextManager:
        context = ContextManager(self.backend, n_ctx=n_ctx, gen_reserve=gen_reserve)
        context.start_session(SYSTEM)
        return context

    def make_history(self, count: int = 10) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for index in range(1, count + 1):
            messages.extend(
                (
                    {"role": "user", "content": user_text(index)},
                    {"role": "assistant", "content": reply_text(index)},
                )
            )
        return messages

    def append_live_turn(self, context: ContextManager, index: int) -> None:
        context.begin_turn(user_text(index))
        context.commit_turn(self.backend.encode(reply_text(index)))

    def test_t1_loaded_history_matches_live_token_log_and_next_prompt(self):
        messages = self.make_history()
        live = self.make_context()
        for index in range(1, 11):
            self.append_live_turn(live, index)

        restored = self.make_context()
        report = restored.load_history(messages)
        self.assertEqual(report.turns_loaded, 10)
        self.assertEqual(report.turns_dropped, 0)
        self.assertEqual(report.messages_skipped, 0)
        self.assertEqual(restored.history_ids, live.history_ids)

        self.assertEqual(
            restored.begin_turn("What should we bring?"),
            live.begin_turn("What should we bring?"),
        )

    def test_t2_new_turns_append_without_changing_loaded_tokens(self):
        restored = self.make_context()
        restored.load_history(self.make_history())
        earlier = restored.history_ids

        for index in range(11, 14):
            self.append_live_turn(restored, index)
            self.assertEqual(restored.history_ids[: len(earlier)], earlier)
            earlier = restored.history_ids

    def test_t3_irregular_sequences_and_skipped_messages(self):
        messages = [
            {"role": "assistant", "content": "An orphaned assistant response."},
            {"role": "user", "content": "First user without a reply."},
            {"role": "user", "content": "Second user has a reply."},
            {"role": "assistant", "content": "The second user's reply."},
            {"role": "user", "content": "Trailing user has no reply."},
            {"role": "tool", "content": "Tool output is not restored."},
            {"role": "system", "content": "System role is not restored."},
            {"role": "assistant", "content": "   "},
        ]
        context = self.make_context()
        report = context.load_history(messages)

        self.assertEqual(report.turns_loaded, 3)
        self.assertEqual(report.turns_dropped, 0)
        self.assertEqual(report.messages_skipped, 3)
        expected = list(context.system_ids)
        expected.extend(context._gen_ids + context._close(self.backend.encode(messages[0]["content"])))
        expected.extend(self.backend.encode(context._user_segment(messages[1]["content"])))
        expected.extend(self.backend.encode(context._user_segment(messages[2]["content"])))
        expected.extend(context._gen_ids + context._close(self.backend.encode(messages[3]["content"])))
        expected.extend(self.backend.encode(context._user_segment(messages[4]["content"])))
        self.assertEqual(context.history_ids, expected)

    def test_t4_truncation_drops_oldest_whole_turns(self):
        messages = self.make_history()
        context = self.make_context(n_ctx=800, gen_reserve=80)
        budget = int(0.5 * (800 - 80))
        report = context.load_history(messages)

        turn_ids = []
        for index in range(1, 11):
            ids = self.backend.encode(context._user_segment(user_text(index)))
            ids.extend(context._gen_ids)
            ids.extend(context._close(self.backend.encode(reply_text(index))))
            turn_ids.append(ids)

        expected_dropped = 0
        expected_tokens = len(context.system_ids) + sum(map(len, turn_ids))
        while expected_tokens > budget:
            expected_tokens -= len(turn_ids[expected_dropped])
            expected_dropped += 1

        self.assertGreater(report.turns_dropped, 0)
        self.assertEqual(report.turns_dropped, expected_dropped)
        self.assertEqual(report.turns_loaded, 10 - expected_dropped)
        self.assertEqual(report.tokens, expected_tokens)
        self.assertEqual(context.token_count(), report.tokens)
        self.assertLessEqual(report.tokens, budget)
        self.assertEqual(
            context.history_ids,
            list(context.system_ids) + [token for turn in turn_ids[expected_dropped:] for token in turn],
        )
        first_kept_user = self.backend.encode(
            context._user_segment(user_text(expected_dropped + 1))
        )
        self.assertEqual(
            context.history_ids[len(context.system_ids) :][: len(first_kept_user)],
            first_kept_user,
        )

    def test_t5_loading_after_open_turn_or_twice_raises(self):
        context = self.make_context()
        context.begin_turn("A pending turn.")
        with self.assertRaises(RuntimeError):
            context.load_history(self.make_history())
        context.abort_turn()

        context.load_history(self.make_history(1))
        with self.assertRaises(RuntimeError):
            context.load_history([])

    def test_t6_empty_history_is_a_no_op(self):
        context = self.make_context()
        before = context.history_ids
        report = context.load_history([])
        self.assertEqual(
            (report.turns_loaded, report.turns_dropped, report.messages_skipped, report.tokens),
            (0, 0, 0, 0),
        )
        self.assertEqual(context.history_ids, before)

    def test_t7_end_of_turn_id_matches_qwen_im_end_token(self):
        context = self.make_context()
        self.assertEqual(context.end_of_turn_id, self.backend.encode("<|im_end|>")[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
