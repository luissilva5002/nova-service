from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nova.config import LLM_MODELS, NOVA_PERSONA_PROMPT
from nova.brain import kv_state
from nova.brain.kv_state import (
    load_snapshot_for,
    restore_snapshot,
    save_snapshot,
    snapshot_from_file,
    snapshot_nbytes,
    snapshot_to_file,
)

MODEL_PATH = LLM_MODELS["qwen2.5-3b"]["path"]
try:
    import llama_cpp

    LLAMA_CPP_AVAILABLE = True
except (ImportError, OSError):
    llama_cpp = None
    LLAMA_CPP_AVAILABLE = False

MODEL_AVAILABLE = MODEL_PATH.is_file()
SKIP_REASON = (
    "Requires llama-cpp-python and the configured Qwen2.5-3B GGUF "
    f"(llama_cpp available={LLAMA_CPP_AVAILABLE}, model exists={MODEL_AVAILABLE})."
)
MAX_GENERATED_TOKENS = 12
FOLLOW_UP = "What did we establish in the previous message?"
FIRST_USER_MESSAGE = "We are planning a calm day in a nearby park."
UNRELATED_USER_MESSAGE = (
    "This is a separate unrelated lane with changing details. "
    "The weather is clear, the train is on time, and the cafe is open. "
) * 30


def greedy_generate(llama, prompt_ids):
    """Generate exactly up to twelve greedy tokens using generate's default reset."""
    generator = llama.generate(prompt_ids, temp=0.0)
    try:
        return list(islice(generator, MAX_GENERATED_TOKENS))
    finally:
        generator.close()


class EvalCounter:
    """Count tokens evaluated before the first generated token is yielded."""

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


def _generate_with_eval_count(llama, prompt_ids) -> tuple[list[int], int]:
    generator = llama.generate(prompt_ids, temp=0.0)
    generated: list[int] = []
    try:
        with EvalCounter(llama) as counter:
            first_token = next(generator)
            counter.stop_counting()
            generated.append(int(first_token))
            generated.extend(
                int(token)
                for token in islice(generator, MAX_GENERATED_TOKENS - 1)
            )
            evaluated_before_first_token = counter.total
    finally:
        generator.close()
    return generated, evaluated_before_first_token


@unittest.skipUnless(LLAMA_CPP_AVAILABLE and MODEL_AVAILABLE, SKIP_REASON)
class KvStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nova.brain.context_manager import ContextManager, LlamaBackend

        cls.ContextManager = ContextManager
        cls.LlamaBackend = LlamaBackend
        cls.llama = llama_cpp.Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_threads=4,
            verbose=False,
        )
        cls.backend = LlamaBackend(cls.llama)
        cls.version = importlib.metadata.version("llama-cpp-python")

    @classmethod
    def tearDownClass(cls):
        cls.llama.close()

    def _new_context(self, system_prompt: str = NOVA_PERSONA_PROMPT):
        context = self.ContextManager(
            self.backend,
            n_ctx=1024,
            gen_reserve=64,
        )
        context.start_session(system_prompt)
        return context

    def _seed_lane(self):
        self.llama.reset()
        context = self._new_context()
        prompt = context.begin_turn(FIRST_USER_MESSAGE)
        generated = greedy_generate(self.llama, prompt)
        context.commit_turn(generated)
        return context, generated

    def _snapshot(self, lane: str = "test"):
        snap = save_snapshot(self.llama, meta={"lane": lane})
        print(
            f"llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}"
        )
        return snap

    def _make_unrelated_prompt(self):
        context = self._new_context("A completely different system prompt.")
        prompt = context.begin_turn(UNRELATED_USER_MESSAGE)
        self.assertGreaterEqual(len(prompt), 100)
        return context, prompt

    def test_t1_round_trip_reuses_lane_prefix_and_matches_control(self):
        lane, _ = self._seed_lane()
        snap = self._snapshot("t1")

        control_prompt = lane.begin_turn(FOLLOW_UP)
        control_ids = greedy_generate(self.llama, control_prompt)
        lane.abort_turn()

        unrelated_lane, unrelated_prompt = self._make_unrelated_prompt()
        greedy_generate(self.llama, unrelated_prompt)
        unrelated_lane.abort_turn()

        restore_snapshot(self.llama, snap)
        replay_prompt = lane.begin_turn(FOLLOW_UP)
        replay_ids, evaluated = _generate_with_eval_count(self.llama, replay_prompt)
        lane.abort_turn()

        print(
            f"T1: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}, "
            f"continuation_prompt_tokens={len(replay_prompt)}, "
            f"evaluated_before_first_generated={evaluated}"
        )
        self.assertEqual(replay_ids, control_ids)
        self.assertLess(evaluated, len(replay_prompt))
        self.assertLessEqual(evaluated, 64)

    def test_t2_snapshot_is_small_compared_with_full_state(self):
        self._seed_lane()
        snap = self._snapshot("t2")
        full_state = self.llama.save_state()
        scores_bytes = int(full_state.scores.nbytes)
        full_state_bytes = scores_bytes + int(full_state.llama_state_size)

        print(
            f"T2: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}, "
            f"save_state_scores_plus_state_bytes={full_state_bytes} "
            f"(scores={scores_bytes}, state={full_state.llama_state_size})"
        )
        self.assertLess(snapshot_nbytes(snap), 64 * 1024 * snap.n_tokens)

    def test_t3_file_round_trip_restores_identical_continuation(self):
        lane, _ = self._seed_lane()
        snap = self._snapshot("t3")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lane.npz"
            snapshot_to_file(snap, path)
            loaded = snapshot_from_file(path, expected_meta={"lane": "t3"})
            self.assertIsNotNone(loaded)
            assert loaded is not None

            control_prompt = lane.begin_turn(FOLLOW_UP)
            control_ids = greedy_generate(self.llama, control_prompt)
            lane.abort_turn()
            _, unrelated_prompt = self._make_unrelated_prompt()
            greedy_generate(self.llama, unrelated_prompt)

            restore_snapshot(self.llama, loaded)
            replay_prompt = lane.begin_turn(FOLLOW_UP)
            replay_ids = greedy_generate(self.llama, replay_prompt)
            lane.abort_turn()

        print(
            f"T3: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}"
        )
        self.assertEqual(replay_ids, control_ids)

    def test_t4_metadata_mismatch_and_corrupt_files_are_rejected(self):
        self._seed_lane()
        snap = self._snapshot("t4")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lane.npz"
            snapshot_to_file(snap, path)
            self.assertIsNone(
                snapshot_from_file(path, expected_meta={"n_ctx": snap.meta["n_ctx"] + 1})
            )

            garbage_path = Path(temp_dir) / "garbage.npz"
            garbage_path.write_bytes(b"not an npz file")
            self.assertIsNone(snapshot_from_file(garbage_path))

            truncated_path = Path(temp_dir) / "truncated.npz"
            truncated_path.write_bytes(path.read_bytes()[:32])
            self.assertIsNone(snapshot_from_file(truncated_path))

        incompatible_llama = SimpleNamespace(
            n_ctx=lambda: snap.meta["n_ctx"] + 1,
            n_vocab=lambda: snap.meta["n_vocab"],
            input_ids=np.zeros(snap.n_tokens, dtype=np.int32),
            n_tokens=0,
            _seed=None,
            model_path=self.llama.model_path,
            context_params=self.llama.context_params,
        )
        with self.assertRaises(ValueError):
            restore_snapshot(incompatible_llama, snap)

        print(
            f"T4: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}"
        )

    def test_t4b_model_identity_mismatches_are_rejected(self):
        self._seed_lane()
        snap = self._snapshot("t4b")
        original_meta = dict(snap.meta)
        tampered_values = {
            "model_fingerprint": "tampered-fingerprint",
            "model_size": snap.meta["model_size"] + 1,
            "llama_cpp_version": "tampered-version",
        }
        for key, value in tampered_values.items():
            with self.subTest(key=key):
                snap.meta[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    restore_snapshot(self.llama, snap)
                snap.meta.clear()
                snap.meta.update(original_meta)

        snap.meta["model_fingerprint"] = tampered_values["model_fingerprint"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tampered.npz"
            snapshot_to_file(snap, path)
            self.assertIsNone(
                load_snapshot_for(self.llama, path, extra_meta={"lane": "t4b"})
            )

        print(
            f"T4b: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}"
        )

    def test_t6_save_and_restore_timing_at_about_600_tokens(self):
        self.llama.reset()
        context = self._new_context("A completely different system prompt.")
        long_message = (
            "This is a separate unrelated lane with changing details. "
            "The weather is clear, the train is on time, and the cafe is open. "
        ) * 18
        prompt = context.begin_turn(long_message)
        greedy_generate(self.llama, prompt)
        context.abort_turn()

        snap = save_snapshot(self.llama, meta={"lane": "t6"})
        self.assertGreaterEqual(snap.n_tokens, 500)
        self.assertLessEqual(snap.n_tokens, 750)

        save_started = time.perf_counter()
        timed_snap = save_snapshot(self.llama, meta={"lane": "t6"})
        save_elapsed = time.perf_counter() - save_started

        self.llama.reset()
        restore_started = time.perf_counter()
        restore_snapshot(self.llama, timed_snap)
        restore_elapsed = time.perf_counter() - restore_started
        save_ms = save_elapsed * 1000
        restore_ms = restore_elapsed * 1000
        print(
            f"T6: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}, "
            f"save_ms={save_ms:.3f}, restore_ms={restore_ms:.3f}"
        )

        if os.environ.get("KV_TIMING_STRICT", "1") != "0":
            self.assertLess(save_elapsed, 1.0)
            self.assertLess(restore_elapsed, 1.0)

    def test_t7_failed_restore_resets_context_and_generation_recovers(self):
        _, _ = self._seed_lane()
        snap = self._snapshot("t7")
        _, unrelated_prompt = self._make_unrelated_prompt()
        greedy_generate(self.llama, unrelated_prompt)

        original_state_api = kv_state._state_api()
        wrong_copy_in = lambda _ctx, _buffer, size: size - 1
        with patch.object(
            kv_state,
            "_state_api",
            return_value=(
                original_state_api[0],
                original_state_api[1],
                wrong_copy_in,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                restore_snapshot(self.llama, snap)

        self.assertEqual(self.llama.n_tokens, 0)
        n_tokens_after_reset = self.llama.n_tokens
        fresh_context = self._new_context()
        fresh_prompt = fresh_context.begin_turn("Can you still answer normally?")
        generated = greedy_generate(self.llama, fresh_prompt)
        fresh_context.abort_turn()
        self.assertTrue(generated)
        print(
            f"T7: llama-cpp-python={self.version}, "
            f"n_tokens_after_reset={n_tokens_after_reset}, "
            f"recovery_generated={len(generated)}"
        )

    @unittest.skipUnless(
        os.environ.get("KV_TEST_TWO_INSTANCES") == "1",
        "Set KV_TEST_TWO_INSTANCES=1 to load the GGUF a second time.",
    )
    def test_t5_restore_file_into_fresh_llama_instance(self):
        lane, first_ids = self._seed_lane()
        snap = self._snapshot("t5")
        control_prompt = lane.begin_turn(FOLLOW_UP)
        control_ids = greedy_generate(self.llama, control_prompt)
        lane.abort_turn()

        fresh_llama = llama_cpp.Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_threads=4,
            verbose=False,
        )
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "lane.npz"
                snapshot_to_file(snap, path)
                loaded = load_snapshot_for(fresh_llama, path, extra_meta={"lane": "t5"})
                self.assertIsNotNone(loaded)
                assert loaded is not None
                restore_snapshot(fresh_llama, loaded)
                fresh_backend = self.LlamaBackend(fresh_llama)
                fresh_lane = self.ContextManager(
                    fresh_backend,
                    n_ctx=1024,
                    gen_reserve=64,
                )
                fresh_lane.start_session(NOVA_PERSONA_PROMPT)
                fresh_lane.begin_turn(FIRST_USER_MESSAGE)
                fresh_lane.commit_turn(first_ids)
                replay_prompt = fresh_lane.begin_turn(FOLLOW_UP)
                replay_ids = greedy_generate(fresh_llama, replay_prompt)
                fresh_lane.abort_turn()
        finally:
            fresh_llama.close()

        print(
            f"T5: llama-cpp-python={self.version}, n_tokens={snap.n_tokens}, "
            f"snapshot_bytes={snapshot_nbytes(snap)}"
        )
        self.assertEqual(replay_ids, control_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
