"""Fake-model tests for action lanes (B2-6) and persisted base states (B2-7).

These run without llama-cpp-python or a GGUF: a small fake llama implements the
parts of the llama.cpp API that lanes touch (token log, prefix reuse, state
save/restore), so the logic is exercised anywhere, including CI.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

_DATA_DIR = tempfile.TemporaryDirectory(prefix="nova-action-lanes-", ignore_cleanup_errors=True)
os.environ.setdefault("NOVA_DATA_DIR", _DATA_DIR.name)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.brain import kv_state  # noqa: E402
from nova.brain import lane_engine as lane_engine_module  # noqa: E402
from nova.brain.generation import GenerationParams  # noqa: E402
from nova.brain.lane_engine import LaneEngine, LaneOverflow  # noqa: E402
from nova.brain.lanes import LaneManager, LaneSpec  # noqa: E402

IM_START, IM_END, EOS = 1, 3, 2
CHAT_PARAMS = GenerationParams(max_tokens=8, temperature=0.0)
PERSONA = "You are Nova, a concise assistant."
MUSIC_TOOLS = [
    {"type": "function", "function": {"name": "ytm_play", "description": "Play a song.",
                                       "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "ytm_pause", "description": "Pause playback.",
                                       "parameters": {"type": "object", "properties": {}}}},
]
CALENDAR_TOOLS = [
    {"type": "function", "function": {"name": "gc_list_today", "description": "List today.",
                                       "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "gc_find", "description": "Find events.",
                                       "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
]
CONTEXT = "[Context]\nCurrent date: Sunday, 2026-11-08 (UTC+00:00)\n[/Context]"


class FakeBackend:
    """ChatML-style renderer and a deterministic word-level tokenizer."""

    SPECIAL = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            if not piece:
                continue
            if piece in self.SPECIAL:
                ids.append(self.SPECIAL[piece])
                continue
            for word in re.findall(r"\S+|\s+", piece):
                ids.append(100 + zlib.crc32(word.encode("utf-8")) % 800)
        return ids

    def render(self, messages, tools, add_generation_prompt):
        out = ""
        for message in messages:
            content = message["content"]
            if message["role"] == "system" and tools:
                content += "\n# Tools\n" + json.dumps(tools, sort_keys=True)
            out += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


class FakeLlama:
    """Just enough llama.cpp: a token log with prefix reuse on generate()."""

    def __init__(self, model_path, n_ctx: int = 1024):
        self.model_path = str(model_path)
        self._n_ctx = n_ctx
        self.input_ids = np.zeros(n_ctx, dtype=np.intc)
        self.n_tokens = 0
        self._seed = None
        self.flash_attn = True
        self.type_k = None
        self.type_v = None
        self._ctx = types.SimpleNamespace(ctx=object())
        self.eval_log: list[list[int]] = []
        self.scripts: list[list[int]] = []
        self.default_script = [10, 11, IM_END]

    def n_ctx(self):
        return self._n_ctx

    def n_vocab(self):
        return 1000

    def token_eos(self):
        return EOS

    def reset(self):
        self.n_tokens = 0

    def eval(self, tokens):
        tokens = [int(token) for token in tokens]
        self.eval_log.append(tokens)
        self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
        self.n_tokens += len(tokens)

    def set_seed(self, seed):
        self._seed = seed

    def detokenize(self, ids):
        return "".join(chr(97 + int(i) % 26) for i in ids).encode("utf-8")

    def generate(self, tokens, **_kwargs):
        tokens = [int(token) for token in tokens]
        common = 0
        for cached, wanted in zip(self.input_ids[:self.n_tokens], tokens):
            if int(cached) != wanted:
                break
            common += 1
        common = min(common, len(tokens) - 1)
        self.n_tokens = common
        self.eval(tokens[common:])
        script = self.scripts.pop(0) if self.scripts else list(self.default_script)
        for token in script:
            yield token
            self.eval([token])


class FakeLlamaTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nova-action-lanes-case-")
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "model.gguf"
        self.model.write_bytes(os.urandom(4096))
        self.base_dir = Path(self.tmp.name) / "kv_base"
        self.llama = FakeLlama(self.model)
        kv_state._MODEL_FINGERPRINTS.clear()

        holder = self

        def fake_state_api():
            def get_size(_ctx):
                return 64 + 4 * int(holder.llama.n_tokens)

            def copy_out(_ctx, buffer, size):
                for index in range(size):
                    buffer[index] = index % 251
                return size

            def copy_in(_ctx, _buffer, size):
                return size

            return get_size, copy_out, copy_in

        patcher = patch.object(kv_state, "_state_api", fake_state_api)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_manager(self, llama=None, base_dir="default", n_ctx=1024):
        directory = self.base_dir if base_dir == "default" else base_dir
        return LaneManager(
            llama or self.llama,
            FakeBackend(),
            n_ctx=n_ctx,
            base_state_dir=directory,
        )

    def base_files(self):
        return sorted(self.base_dir.glob("*.npz")) if self.base_dir.exists() else []

    @staticmethod
    def music_spec(tools=None):
        return LaneSpec(name="action_music", system_prompt=PERSONA, tools=tools or MUSIC_TOOLS, gen_reserve=64)


class BaseStateTests(FakeLlamaTestCase):
    def test_first_warm_builds_and_persists(self):
        manager = self.make_manager()
        lane = manager.action_lane(self.music_spec())
        prefilled = manager.warm(lane)

        system_ids = lane.context.system_ids
        self.assertEqual(prefilled, len(system_ids))
        self.assertEqual(self.llama.eval_log, [system_ids])
        stats = manager.stats()
        self.assertEqual(stats["cold_starts"], 1)
        self.assertEqual(stats["base_saved"], 1)
        self.assertEqual(stats["base_loaded"], 0)
        files = self.base_files()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].name.startswith("action_music-"))
        self.assertIn("-ctx1024", files[0].name)

    def test_new_manager_restores_from_disk_without_prefill(self):
        first = self.make_manager()
        first.warm(first.action_lane(self.music_spec()))
        expected_ids = first.action_lane(self.music_spec()).context.system_ids

        fresh_llama = FakeLlama(self.model)
        holder_llama = self.llama
        self.llama = fresh_llama  # the fake state API follows the active llama
        try:
            second = self.make_manager(llama=fresh_llama)
            lane = second.action_lane(self.music_spec())
            prefilled = second.warm(lane)
        finally:
            self.llama = holder_llama

        self.assertEqual(prefilled, 0)
        self.assertEqual(fresh_llama.eval_log, [])
        self.assertEqual(fresh_llama.n_tokens, len(expected_ids))
        self.assertEqual([int(t) for t in fresh_llama.input_ids[:fresh_llama.n_tokens]], expected_ids)
        stats = second.stats()
        self.assertEqual(stats["base_loaded"], 1)
        self.assertEqual(stats["cold_starts"], 0)
        self.assertEqual(stats["warm_prefill_tokens"], 0)
        self.assertEqual(len(self.base_files()), 1)

    def test_changed_tools_get_a_new_file_and_old_one_is_pruned(self):
        manager = self.make_manager()
        manager.warm(manager.action_lane(self.music_spec()))
        old_files = self.base_files()

        other = self.make_manager()
        other.warm(other.action_lane(self.music_spec(CALENDAR_TOOLS)))
        files = self.base_files()
        self.assertEqual(len(files), 1)
        self.assertNotEqual(files[0], old_files[0])

    def test_corrupt_file_falls_back_to_rebuild_and_is_rewritten(self):
        first = self.make_manager()
        first.warm(first.action_lane(self.music_spec()))
        (path,) = self.base_files()
        path.write_bytes(b"not an npz archive")

        fresh = FakeLlama(self.model)
        self.llama = fresh
        second = self.make_manager(llama=fresh)
        lane = second.action_lane(self.music_spec())
        prefilled = second.warm(lane)

        self.assertEqual(prefilled, len(lane.context.system_ids))
        stats = second.stats()
        self.assertEqual(stats["base_invalid"], 1)
        self.assertEqual(stats["base_saved"], 1)
        self.assertGreater(path.stat().st_size, 100)  # a valid archive again

    def test_changed_model_invalidates_the_state(self):
        first = self.make_manager()
        first.warm(first.action_lane(self.music_spec()))

        self.model.write_bytes(os.urandom(8192))  # different size => different identity
        kv_state._MODEL_FINGERPRINTS.clear()
        fresh = FakeLlama(self.model)
        self.llama = fresh
        second = self.make_manager(llama=fresh)
        lane = second.action_lane(self.music_spec())
        prefilled = second.warm(lane)

        self.assertEqual(prefilled, len(lane.context.system_ids))
        self.assertEqual(second.stats()["base_invalid"], 1)
        self.assertEqual(second.stats()["base_loaded"], 0)

    def test_no_base_dir_keeps_previous_behaviour(self):
        manager = self.make_manager(base_dir=None)
        lane = manager.action_lane(self.music_spec())
        self.assertEqual(manager.warm(lane), len(lane.context.system_ids))
        self.assertEqual(self.base_files(), [])
        stats = manager.stats()
        self.assertEqual((stats["base_saved"], stats["base_loaded"], stats["base_invalid"]), (0, 0, 0))

    def test_unwritable_directory_does_not_break_warming(self):
        blocker = Path(self.tmp.name) / "blocker"
        blocker.write_text("a file, not a directory")
        manager = self.make_manager(base_dir=blocker / "kv_base")
        lane = manager.action_lane(self.music_spec())
        self.assertEqual(manager.warm(lane), len(lane.context.system_ids))
        self.assertEqual(manager.stats()["base_save_failures"], 1)
        self.assertTrue(lane.warm)

    def test_forget_action_drops_lane_and_snapshot(self):
        manager = self.make_manager()
        lane = manager.action_lane(self.music_spec())
        manager.warm(lane)
        manager.forget_action("action_music")
        self.assertEqual(manager.stats()["snapshots"], 0)
        self.assertIsNone(manager.stats()["resident"])
        with self.assertRaises(ValueError):
            manager.warm(lane)  # no longer owned by the manager


class ActionTurnTests(FakeLlamaTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(lane_engine_module, "LlamaBackend", lambda _llama: FakeBackend())
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_engine(self, llama=None, base_dir="default", n_ctx=1024):
        directory = self.base_dir if base_dir == "default" else base_dir
        return LaneEngine(
            llama or self.llama,
            n_ctx,
            PERSONA,
            gen_reserve=64,
            base_state_dir=directory,
        )

    def test_action_turn_generates_and_keeps_the_lane_clean(self):
        engine = self.make_engine()
        self.llama.scripts.append([21, 22, 23, IM_END])
        result = engine.action_turn("action_music", MUSIC_TOOLS, "play some jazz", CONTEXT, CHAT_PARAMS)

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.ids, [21, 22, 23, IM_END])
        self.assertEqual(result.metrics["completion_tokens"], 3)
        self.assertIn("action_lane", result.metrics)
        self.assertNotIn("lane", result.metrics)  # main.py reads metrics["lane"] as "chat lane handled it"
        self.assertFalse(result.committed)

        lane = engine._action_lane("action_music", MUSIC_TOOLS)
        self.assertEqual(lane.context.history_ids, lane.context.system_ids)
        self.assertEqual(engine.stats()["action_turns"], 1)

    def test_second_action_turn_reuses_the_prefix(self):
        engine = self.make_engine()
        first = engine.action_turn("action_music", MUSIC_TOOLS, "play some jazz", CONTEXT, CHAT_PARAMS)
        system_len = len(engine._action_lane("action_music", MUSIC_TOOLS).context.system_ids)
        second = engine.action_turn("action_music", MUSIC_TOOLS, "pause the music", CONTEXT, CHAT_PARAMS)

        self.assertEqual(first.metrics["warm_prefill_tokens"], system_len)  # first turn built the prefix
        self.assertEqual(second.metrics["warm_prefill_tokens"], 0)
        self.assertGreaterEqual(second.metrics["cache_hit_tokens"], system_len)
        self.assertLess(second.metrics["evaluated_tokens"], first.metrics["prompt_tokens"])

    def test_action_turn_after_a_chat_turn_restores_the_base_snapshot(self):
        engine = self.make_engine()
        engine.action_turn("action_music", MUSIC_TOOLS, "play some jazz", CONTEXT, CHAT_PARAMS)
        system_len = len(engine._action_lane("action_music", MUSIC_TOOLS).context.system_ids)

        engine.chat("session-1", "hello there", CONTEXT, [], CHAT_PARAMS)  # swaps the chat lane in
        swaps_before = engine.stats()["swaps_in"]
        result = engine.action_turn("action_music", MUSIC_TOOLS, "pause the music", CONTEXT, CHAT_PARAMS)

        self.assertEqual(engine.stats()["swaps_in"], swaps_before + 1)
        self.assertEqual(result.metrics["cache_hit_tokens"], system_len)
        self.assertEqual(result.metrics["warm_prefill_tokens"], 0)  # restored, not rebuilt
        # ...and the chat lane is still intact afterwards.
        chat_again = engine.chat("session-1", "and again", CONTEXT, [], CHAT_PARAMS)
        self.assertEqual(chat_again.finish_reason, "stop")

    def test_changed_tools_replace_the_lane(self):
        engine = self.make_engine()
        engine.action_turn("action_music", MUSIC_TOOLS, "play jazz", CONTEXT, CHAT_PARAMS)
        old_lane = engine._action_lane("action_music", MUSIC_TOOLS)
        changed = MUSIC_TOOLS + [CALENDAR_TOOLS[0]]
        engine.action_turn("action_music", changed, "play jazz", CONTEXT, CHAT_PARAMS)
        new_lane = engine._action_lane("action_music", changed)
        self.assertIsNot(old_lane, new_lane)
        self.assertGreater(len(new_lane.context.system_ids), len(old_lane.context.system_ids))

    def test_warm_action_reports_its_source(self):
        engine = self.make_engine()
        built = engine.warm_action("action_music", MUSIC_TOOLS)
        self.assertEqual(built["source"], "built")
        self.assertGreater(built["prefilled_tokens"], 0)
        self.assertEqual(engine.warm_action("action_music", MUSIC_TOOLS)["source"], "memory")

        fresh = FakeLlama(self.model)
        self.llama = fresh
        restored = self.make_engine(llama=fresh).warm_action("action_music", MUSIC_TOOLS)
        self.assertEqual(restored["source"], "disk")
        self.assertEqual(restored["prefilled_tokens"], 0)
        self.assertEqual(fresh.eval_log, [])

    def test_oversized_turn_raises_lane_overflow(self):
        engine = self.make_engine(n_ctx=400)
        huge = "word " * 600
        with self.assertRaises(LaneOverflow):
            engine.action_turn("action_music", MUSIC_TOOLS, huge, CONTEXT, CHAT_PARAMS)
        self.assertEqual(engine.stats()["overflows"], 1)

    def test_independent_action_lanes_do_not_share_prefixes(self):
        engine = self.make_engine()
        engine.warm_action("action_music", MUSIC_TOOLS)
        engine.warm_action("action_calendar", CALENDAR_TOOLS)
        self.assertEqual(len(self.base_files()), 2)
        self.assertEqual(engine.stats()["snapshots"], 2)


if __name__ == "__main__":
    unittest.main()
