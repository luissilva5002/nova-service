"""Token-ID generation and incremental text decoding for llama.cpp."""
from __future__ import annotations

import codecs
import inspect
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

try:
    from llama_cpp import Llama

    _CHAT_COMPLETION_PARAMETERS = inspect.signature(Llama.create_chat_completion).parameters
except (ImportError, AttributeError, ValueError):
    _CHAT_COMPLETION_PARAMETERS = {}


def _completion_default(name: str, fallback):
    parameter = _CHAT_COMPLETION_PARAMETERS.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return fallback
    return parameter.default


@dataclass
class GenerationParams:
    max_tokens: int
    temperature: float = _completion_default("temperature", 0.2)
    top_p: float = _completion_default("top_p", 0.95)
    top_k: int = _completion_default("top_k", 40)
    min_p: float = _completion_default("min_p", 0.05)
    typical_p: float = _completion_default("typical_p", 1.0)
    repeat_penalty: float = _completion_default("repeat_penalty", 1.0)
    frequency_penalty: float = _completion_default("frequency_penalty", 0.0)
    presence_penalty: float = _completion_default("presence_penalty", 0.0)
    seed: int | None = _completion_default("seed", None)


class Utf8StreamDecoder:
    """Incrementally decode token bytes while retaining incomplete UTF-8."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, data: bytes) -> str:
        return self._decoder.decode(data, final=False)

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)


@dataclass
class GenerationResult:
    ids: list[int]
    text: str
    finish_reason: Literal["stop", "length", "cancelled"]
    prompt_tokens: int
    reused_tokens: int
    evaluated_tokens: int
    generated_tokens: int


class TokenGenerator:
    def __init__(self, llama, stop_ids: set[int]):
        self._llama = llama
        self._stop_ids = set(stop_ids)

    def stream(
        self,
        prompt_ids: list[int],
        params: GenerationParams,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str | GenerationResult]:
        prompt = [int(token_id) for token_id in prompt_ids]
        prompt_tokens = len(prompt)
        n_ctx = int(self._llama.n_ctx())
        if not prompt:
            raise ValueError("prompt_ids must not be empty.")
        if prompt_tokens >= n_ctx:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens and must be shorter than n_ctx={n_ctx}."
            )
        if params.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        active_count = int(self._llama.n_tokens)
        active_ids = list(self._llama.input_ids[:active_count])
        common_prefix = 0
        for cached_id, prompt_id in zip(active_ids, prompt):
            if int(cached_id) != prompt_id:
                break
            common_prefix += 1
        reused_tokens = common_prefix
        if common_prefix == prompt_tokens:
            reused_tokens = max(0, reused_tokens - 1)
        evaluated_tokens = prompt_tokens - reused_tokens
        max_tokens = min(params.max_tokens, n_ctx - prompt_tokens)

        if params.seed is not None:
            self._llama.set_seed(params.seed)

        token_stream = None
        decoder = Utf8StreamDecoder()
        generated_ids: list[int] = []
        text_parts: list[str] = []
        finish_reason = "length"
        kwargs = {
            "top_k": params.top_k,
            "top_p": params.top_p,
            "min_p": params.min_p,
            "typical_p": params.typical_p,
            "temp": params.temperature,
            "repeat_penalty": params.repeat_penalty,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        }

        try:
            # Keep reset at llama.cpp's default: reset=False disables prefix reuse.
            token_stream = self._llama.generate(prompt, **kwargs)
            for token in token_stream:
                token_id = int(token)
                generated_ids.append(token_id)
                if token_id in self._stop_ids:
                    finish_reason = "stop"
                    break

                delta = decoder.feed(self._llama.detokenize([token_id]))
                if delta:
                    text_parts.append(delta)
                    yield delta

                if should_stop is not None and should_stop():
                    finish_reason = "cancelled"
                    break
                if len(generated_ids) >= max_tokens:
                    finish_reason = "length"
                    break

            tail = decoder.flush()
            if tail:
                text_parts.append(tail)
                yield tail

            yield GenerationResult(
                ids=generated_ids,
                text="".join(text_parts),
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                reused_tokens=reused_tokens,
                evaluated_tokens=evaluated_tokens,
                generated_tokens=len(generated_ids),
            )
        finally:
            if token_stream is not None:
                token_stream.close()

    def generate(
        self,
        prompt_ids: list[int],
        params: GenerationParams,
        should_stop: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        result: GenerationResult | None = None
        for item in self.stream(prompt_ids, params, should_stop):
            if isinstance(item, GenerationResult):
                result = item
        if result is None:
            raise RuntimeError("Token generation ended without a result.")
        return result


def stop_ids_for(context_manager, llama) -> set[int]:
    stop_ids = {int(context_manager.end_of_turn_id)}
    eos_id = int(llama.token_eos())
    if eos_id >= 0:
        stop_ids.add(eos_id)
    return stop_ids
