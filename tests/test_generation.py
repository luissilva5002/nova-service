"""Tests for token-ID generation. Run: python tests/test_generation.py -v"""
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
    Utf8StreamDecoder,
    stop_ids_for,
)

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
SYSTEM = "You are a concise assistant. Follow the user's response format exactly."


@unittest.skipUnless(LLAMA_AVAILABLE and MODEL_PATH.is_file(), SKIP_REASON)
class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_threads=4,
            verbose=False,
        )
        cls.backend = LlamaBackend(cls.llama)
        cls.stop_ids = stop_ids_for(
            cls._make_context_for(cls.backend),
            cls.llama,
        )
        cls.generator = TokenGenerator(cls.llama, cls.stop_ids)

    @classmethod
    def tearDownClass(cls):
        cls.llama.close()

    @staticmethod
    def _make_context_for(backend):
        context = ContextManager(backend, n_ctx=1024, gen_reserve=64)
        context.start_session(SYSTEM)
        return context

    def setUp(self):
        self.llama.reset()

    def make_context(self) -> ContextManager:
        return self._make_context_for(self.backend)

    def test_t1_greedy_text_matches_chat_completion(self):
        user_text = "Reply with exactly one word: hello"
        context = self.make_context()
        prompt = context.begin_turn(user_text)
        result = self.generator.generate(
            prompt,
            GenerationParams(max_tokens=40, temperature=0.0),
        )
        completion = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=40,
        )
        completion_text = completion["choices"][0]["message"]["content"].strip()
        generated_text = result.text.strip()
        if generated_text != completion_text:
            print(f"T1 TokenGenerator: {generated_text!r}")
            print(f"T1 chat completion: {completion_text!r}")
        self.assertEqual(generated_text, completion_text)

    def test_t2_stop_token_is_kept_in_ids_but_not_text(self):
        context = self.make_context()
        prompt = context.begin_turn("Reply with exactly one word: OK")
        result = self.generator.generate(
            prompt,
            GenerationParams(max_tokens=40, temperature=0.0),
        )
        self.assertEqual(result.finish_reason, "stop")
        self.assertIn(result.ids[-1], self.stop_ids)
        self.assertNotIn("<|im_end|>", result.text)

    def test_t3_max_tokens_finishes_by_length(self):
        context = self.make_context()
        prompt = context.begin_turn("Write a detailed explanation of a garden.")
        result = self.generator.generate(
            prompt,
            GenerationParams(max_tokens=5, temperature=0.0),
        )
        self.assertEqual(len(result.ids), 5)
        self.assertEqual(result.generated_tokens, 5)
        self.assertEqual(result.finish_reason, "length")

    def test_t4_incremental_utf8_decoder_holds_incomplete_sequences(self):
        decoder = Utf8StreamDecoder()
        emoji = "🙂".encode("utf-8")
        self.assertEqual(decoder.feed(emoji[:1]), "")
        self.assertEqual(decoder.feed(emoji[1:3]), "")
        self.assertEqual(decoder.feed(emoji[3:]), "🙂")
        self.assertEqual(decoder.flush(), "")

        decoder = Utf8StreamDecoder()
        accented = "é".encode("utf-8")
        self.assertEqual(decoder.feed(accented[:1]), "")
        self.assertEqual(decoder.feed(accented[1:]), "é")
        self.assertEqual(decoder.flush(), "")

    def test_t5_cancellation_closes_stream_and_followup_reuses_prompt(self):
        context = self.make_context()
        prompt = context.begin_turn("Explain how trees grow.")
        checks = 0

        def should_stop():
            nonlocal checks
            checks += 1
            return checks >= 3

        cancelled = self.generator.generate(
            prompt,
            GenerationParams(max_tokens=40, temperature=0.0),
            should_stop=should_stop,
        )
        self.assertEqual(cancelled.finish_reason, "cancelled")
        self.assertEqual(len(cancelled.ids), 3)
        context.commit_turn(cancelled.ids)

        next_prompt = context.begin_turn("Continue with one short sentence.")
        next_result = self.generator.generate(
            next_prompt,
            GenerationParams(max_tokens=8, temperature=0.0),
        )
        self.assertTrue(next_result.ids)
        self.assertGreaterEqual(next_result.reused_tokens, len(prompt))

    def test_t6_reuse_accounting_matches_prompt_eval_count(self):
        context = self.make_context()
        first_prompt = context.begin_turn("We are planning a visit to the park.")
        first = self.generator.generate(
            first_prompt,
            GenerationParams(max_tokens=8, temperature=0.0),
        )
        context.commit_turn(first.ids)

        next_prompt = context.begin_turn("What should we bring?")
        counted_generate = self.llama.generate
        actual_eval_count = 0
        counting = True
        original_eval = self.llama.eval

        def counted_eval(tokens, *args, **kwargs):
            nonlocal actual_eval_count
            if counting:
                actual_eval_count += len(tokens)
            return original_eval(tokens, *args, **kwargs)

        def counting_generate(tokens, *args, **kwargs):
            nonlocal counting
            token_stream = counted_generate(tokens, *args, **kwargs)
            try:
                for token in token_stream:
                    counting = False
                    yield token
            finally:
                token_stream.close()

        self.llama.eval = counted_eval
        self.llama.generate = counting_generate
        try:
            second = self.generator.generate(
                next_prompt,
                GenerationParams(max_tokens=8, temperature=0.0),
            )
        finally:
            del self.llama.eval
            del self.llama.generate

        self.assertGreaterEqual(second.reused_tokens, len(first_prompt))
        self.assertEqual(
            second.evaluated_tokens,
            len(next_prompt) - second.reused_tokens,
        )
        self.assertLessEqual(abs(second.evaluated_tokens - actual_eval_count), 1)

    def test_t7_sampling_parameters_reach_llama_generate(self):
        context = self.make_context()
        prompt = context.begin_turn("Say a short greeting.")
        params = GenerationParams(
            max_tokens=8,
            temperature=0.37,
            top_k=17,
            top_p=0.81,
            min_p=0.12,
            repeat_penalty=1.23,
        )
        original_generate = self.llama.generate
        captured = {}

        def capture(tokens, *args, **kwargs):
            captured.update(kwargs)
            return original_generate(tokens, *args, **kwargs)

        with patch.object(self.llama, "generate", side_effect=capture):
            self.generator.generate(prompt, params)

        self.assertEqual(captured["temp"], params.temperature)
        self.assertEqual(captured["top_k"], params.top_k)
        self.assertEqual(captured["top_p"], params.top_p)
        self.assertEqual(captured["min_p"], params.min_p)
        self.assertEqual(captured["repeat_penalty"], params.repeat_penalty)
        self.assertNotIn("reset", captured)

    def test_t8_prompt_validation_and_max_tokens_clamp(self):
        with self.assertRaises(ValueError):
            self.generator.generate([], GenerationParams(max_tokens=5))

        context = self.make_context()
        prompt = context.begin_turn("A short prompt.")
        with patch.object(self.llama, "n_ctx", return_value=len(prompt)):
            with self.assertRaises(ValueError):
                self.generator.generate(prompt, GenerationParams(max_tokens=5))

        token_id = next(
            token
            for token in self.backend.encode(" synthetic")
            if token not in self.stop_ids
        )
        captured_tokens = []

        def synthetic_generate(tokens, **kwargs):
            captured_tokens.append(list(tokens))
            yield from [token_id] * 10

        with (
            patch.object(self.llama, "n_ctx", return_value=len(prompt) + 3),
            patch.object(self.llama, "generate", side_effect=synthetic_generate),
        ):
            clamped = self.generator.generate(
                prompt,
                GenerationParams(max_tokens=20, temperature=0.0),
            )
        self.assertEqual(clamped.generated_tokens, 3)
        self.assertEqual(clamped.finish_reason, "length")
        self.assertEqual(captured_tokens, [prompt])

    def test_stream_emits_nonempty_text_and_one_final_result(self):
        context = self.make_context()
        prompt = context.begin_turn("Say hello in one word.")
        items = list(
            self.generator.stream(
                prompt,
                GenerationParams(max_tokens=12, temperature=0.0),
            )
        )
        self.assertTrue(items)
        self.assertIsInstance(items[-1], GenerationResult)
        self.assertEqual(sum(isinstance(item, GenerationResult) for item in items), 1)
        self.assertTrue(all(item for item in items[:-1]))
        self.assertTrue(all(isinstance(item, str) for item in items[:-1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
