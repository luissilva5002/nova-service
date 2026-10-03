"""Tests for nova/brain/context_manager.py. Run: python tests/test_context_manager.py -v

Loads only the GGUF vocabulary and chat template (vocab_only), not the model weights.
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NOVA_LLM_MODEL", "qwen2.5-3b")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.brain.context_manager import ContextManager, ContextOverflow  # noqa: E402

SYSTEM = "You are Nova, a concise personal assistant."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
BLOCK = "[Retrieved context]\n" + "\n".join(
    f"Note {i}: local project context is versioned, searchable, and kept on device."
    for i in range(1, 12)
)
TOOL_CALL = '<tool_call>\n{"name": "memory_search", "arguments": {"query": "battery"}}\n</tool_call>'

_BACKEND = None


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        from llama_cpp import Llama
        from nova.brain.context_manager import LlamaBackend
        from nova.config import LLM_MODEL_PATH

        llm = Llama(model_path=str(LLM_MODEL_PATH), vocab_only=True, verbose=False)
        _BACKEND = LlamaBackend(llm)
    return _BACKEND


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def user_text(i):
    return f"Message {i}: " + "we are planning a relaxed Saturday outing with a walk and lunch. " * (1 + i % 3)


def reply_text(i):
    return f"Reply {i}: sounds good, a walk first and lunch after."


class Base(unittest.TestCase):
    def setUp(self):
        self.b = get_backend()
        self.cm = ContextManager(self.b, n_ctx=4096, gen_reserve=256)
        self.cm.start_session(SYSTEM, TOOLS)

    def run_turns(self, n, blocks=()):
        """Simulate n turns. Returns list of (prompt_ids, reply_ids)."""
        records = []
        for i in range(1, n + 1):
            prompt = self.cm.begin_turn(user_text(i), BLOCK if i in blocks else None)
            reply = self.b.encode(reply_text(i))
            self.cm.commit_turn(reply)
            records.append((prompt, reply))
        return records


class T0TemplateMode(Base):
    def test_report_template_mode(self):
        print(f"\nTEMPLATE MODE: {self.cm.template_mode}")
        if self.cm.template_note:
            print(f"TEMPLATE NOTE: {self.cm.template_note}")
        print(f"USER WRAP: {self.cm._wrap['user']!r}")
        print(f"TOOL WRAP: {self.cm._wrap['tool']!r}")
        print(f"SUFFIX   : {self.cm._suffix_text!r}")


class T1NoLag(Base):
    def test_full_reuse_without_context_blocks(self):
        records = self.run_turns(12)
        for n in range(len(records) - 1):
            prompt, reply = records[n]
            nxt_prompt, _ = records[n + 1]
            cached = prompt + reply
            shared = common_prefix(nxt_prompt, cached)
            self.assertEqual(shared, len(cached), f"turn {n + 2}: reuse {shared} of {len(cached)}")
        print("\nT1 table: turn | prompt tokens | shared with previous prompt+reply")
        for n, (prompt, _) in enumerate(records, 1):
            prev = records[n - 2][0] + records[n - 2][1] if n > 1 else []
            print(f"  {n:>2} | {len(prompt):>5} | {common_prefix(prompt, prev):>5}")


class T2Snippets(Base):
    def test_snippets_not_kept_and_divergence_at_block_start(self):
        blocks = {4, 6, 8, 10}
        records = []
        expected_len = len(self.cm.system_ids)
        for i in range(1, 13):
            hist_before = len(self.cm.history_ids)
            use_block = BLOCK if i in blocks else None
            prompt = self.cm.begin_turn(user_text(i), use_block)
            reply = self.b.encode(reply_text(i))
            self.cm.commit_turn(reply)
            records.append((prompt, reply, hist_before, i in blocks))
            clean = self.b.encode(self.cm._user_segment(user_text(i)))
            expected_len += len(clean) + len(self.cm._gen_ids) + len(self.cm._close(reply))
            self.assertEqual(self.cm.token_count(), expected_len, f"history holds extra tokens after turn {i}")
        for n in range(len(records) - 1):
            prompt, reply, hist_before, had_block = records[n]
            nxt_prompt = records[n + 1][0]
            shared = common_prefix(nxt_prompt, prompt + reply)
            if had_block:
                clean = self.b.encode(self.cm._user_segment(user_text(n + 1)))
                shown = self.b.encode(self.cm._user_segment(user_text(n + 1) + "\n\n" + BLOCK))
                expected = hist_before + common_prefix(clean, shown)
                self.assertEqual(shared, expected, f"turn {n + 2}: divergence not at block start")
            else:
                self.assertEqual(shared, len(prompt) + len(reply))


class T3ToolRoundTrip(Base):
    def test_prefix_survives_tool_call(self):
        self.run_turns(2)
        p1 = self.cm.begin_turn("What connects the PCB to the battery?")
        call_ids = self.b.encode(TOOL_CALL)
        p2 = self.cm.continue_after_tool(call_ids, "The PCB connects to the battery with pogo pins.")
        self.assertEqual(common_prefix(p2, p1 + call_ids), len(p1) + len(call_ids))
        sys_len = len(self.cm.system_ids)
        self.assertEqual(p1[:sys_len], p2[:sys_len])
        final = self.b.encode("Pogo pins.")
        self.cm.commit_turn(final)
        p3 = self.cm.begin_turn("How many?")
        self.assertEqual(common_prefix(p3, p2 + final), len(p2) + len(final))

    def test_tool_segment_shape(self):
        seg = self.cm._tool_segment("RESULT")
        self.assertTrue(seg.startswith("<|im_start|>user\n<tool_response>\n"), repr(seg))
        self.assertTrue(seg.endswith("</tool_response><|im_end|>\n"), repr(seg))


class T4Equivalence(Base):
    def test_matches_full_template_render(self):
        messages = [{"role": "system", "content": SYSTEM}]
        for i in range(1, 4):
            self.cm.begin_turn(user_text(i))
            reply = reply_text(i)
            self.cm.commit_turn(self.b.encode(reply))
            messages += [
                {"role": "user", "content": user_text(i)},
                {"role": "assistant", "content": reply},
            ]
        ours = self.cm.begin_turn(user_text(4))
        messages.append({"role": "user", "content": user_text(4)})
        theirs = self.b.encode(self.b.render(messages, TOOLS, True))
        if ours != theirs:
            k = common_prefix(ours, theirs)
            dec = getattr(self.b, "decode", None)
            if dec:
                print("\nOURS  :", repr(dec(ours[max(0, k - 8): k + 8])))
                print("THEIRS:", repr(dec(theirs[max(0, k - 8): k + 8])))
            print(f"first difference at index {k}; lengths ours={len(ours)} theirs={len(theirs)}")
        self.assertEqual(ours, theirs)


class T5NoDropAndCompaction(unittest.TestCase):
    def test_nothing_dropped_and_flag_flips(self):
        b = get_backend()
        cm = ContextManager(b, n_ctx=2048, gen_reserve=48)
        cm.start_session(SYSTEM, TOOLS)
        threshold = int(0.75 * (2048 - 48))
        first_flip = None
        previous = cm.history_ids
        for i in range(1, 21):
            cm.begin_turn(user_text(i) + " extra words " * 10)
            cm.commit_turn(b.encode(reply_text(i) + " more words " * 5))
            current = cm.history_ids
            self.assertEqual(current[: len(previous)], previous, f"history changed at turn {i}")
            self.assertGreater(len(current), len(previous))
            previous = current
            flag = cm.needs_compaction()
            self.assertEqual(flag, cm.token_count() >= threshold)
            if flag and first_flip is None:
                first_flip = i
                break
        self.assertIsNotNone(first_flip, "needs_compaction never became True in 20 turns")

    def test_overflow_raises_and_leaves_state_clean(self):
        b = get_backend()
        cm = ContextManager(b, n_ctx=512, gen_reserve=64)
        cm.start_session(SYSTEM, TOOLS)
        before = cm.history_ids
        with self.assertRaises(ContextOverflow):
            cm.begin_turn("word " * 1000)
        self.assertEqual(cm.history_ids, before)
        cm.begin_turn("short")  # still usable


class T6ExactIds(Base):
    def test_commit_stores_given_ids(self):
        self.cm.begin_turn("Hi")
        ids = self.b.encode("Hello there")
        self.cm.commit_turn(ids)
        tail = self.cm.history_ids[-(len(ids) + len(self.cm._suffix_ids)):]
        self.assertEqual(tail, ids + self.cm._suffix_ids)

    def test_no_duplicate_end_token(self):
        self.cm.begin_turn("Hi")
        ids = self.b.encode("Hello there") + [self.cm._suffix_ids[0]]
        self.cm.commit_turn(ids)
        tail = self.cm.history_ids[-len(ids) - len(self.cm._suffix_ids) + 1:]
        self.assertEqual(tail, ids + self.cm._suffix_ids[1:])


if __name__ == "__main__":
    unittest.main()