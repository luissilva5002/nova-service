"""
nova/brain/llm_engine.py

The Brain - a single local LLM (Qwen 2.5 3B or Llama 3.2 3B, GGUF,
Q4_K_M) loaded once via llama-cpp-python. Handles:
  - persona/system prompt prefix (KV-cached so it's paid for once)
  - native tool/function calling against the schemas collected by
    the ToolRouter
  - plain conversational generation when no tool applies

This is intentionally the ONLY LLM in the system (see architecture
doc "Anatomy of NOVA's Single-Brain System") - whisper.cpp and Piper
are non-LLM "organs", not separate brains.
"""
import datetime
import asyncio
import json
import logging
import time
import gc
import re
import threading
import queue
from typing import Optional, AsyncIterator

from nova.config import (
    LLM_MODEL_PATH,
    LLM_MODELS,
    LLM_DEFAULT_MODEL_ID,
    LLM_CONTEXT_SIZE,
    LLM_THREADS,
    LLM_THREADS_BATCH,
    LLM_BATCH,
    LLM_FLASH_ATTN,
    LLM_MLOCK,
    LLM_KV_CACHE_TYPE,
    LLM_CACHE_MB,
    LLM_TEMP_TOOLS,
    LLM_MAX_TOKENS_CHAT,
    LLM_MAX_TOKENS_TOOL,
    LLM_GPU_LAYERS,
    NOVA_PERSONA_PROMPT,
)

logger = logging.getLogger("nova.llm_engine")
_THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)

try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:  # pragma: no cover - lets the server boot before the wheel/model is installed
    _LLAMA_CPP_AVAILABLE = False


class LLMEngine:
    def __init__(self):
        self._llm: Optional["Llama"] = None
        self._loaded = False
        self._active_model_id = LLM_DEFAULT_MODEL_ID
        self._lock = asyncio.Lock()

    def list_models(self) -> list[dict]:
        return [
            {
                "id": model_id,
                "label": spec["label"],
                "installed": spec["path"].exists(),
                "active": model_id == self._active_model_id,
            }
            for model_id, spec in LLM_MODELS.items()
        ]

    def select_model(self, model_id: str) -> dict:
        if model_id not in LLM_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        model_path = LLM_MODELS[model_id]["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model is not installed: {model_path.name}")
        if model_id == self._active_model_id and self._llm is not None:
            return {"active_model_id": model_id, "label": LLM_MODELS[model_id]["label"]}

        previous_model_id = self._active_model_id
        logger.info("Switching brain model to %s ...", LLM_MODELS[model_id]["label"])
        if self._llm is not None and hasattr(self._llm, "close"):
            self._llm.close()
        self._llm = None
        self._loaded = False
        gc.collect()
        self._active_model_id = model_id
        try:
            self.load()
            if self._llm is None:
                raise RuntimeError(f"Failed to load model: {model_path.name}")
        except Exception:
            logger.exception("Model switch failed; restoring %s.", LLM_MODELS[previous_model_id]["label"])
            self._llm = None
            self._loaded = False
            self._active_model_id = previous_model_id
            self.load()
            raise
        return {"active_model_id": model_id, "label": LLM_MODELS[model_id]["label"]}

    def _configure_qwen3_non_thinking_mode(self) -> None:
        """Disable Qwen3 reasoning through its Jinja chat-template argument.

        The installed llama-cpp-python API cannot pass template keyword
        arguments per request, so wrap the selected model's chat handler once
        at load time. This prevents reasoning tokens from being generated.
        """
        if self._llm is None or not re.match(r"^qwen3(\.\d+)?-", self._active_model_id):
            return
        try:
            from llama_cpp import llama_chat_format

            base_handler = self._llm.chat_handler
            if base_handler is None:
                chat_format = self._llm.chat_format
                base_handler = self._llm._chat_handlers.get(chat_format)  # type: ignore[attr-defined]
            if base_handler is None:
                logger.warning("Could not find Qwen3 chat handler; reasoning may remain enabled.")
                return

            def non_thinking_handler(*args, **kwargs):
                kwargs["enable_thinking"] = False
                return base_handler(*args, **kwargs)

            self._llm.chat_handler = non_thinking_handler
            logger.info("Disabled Qwen3 thinking mode.")
        except Exception:  # pragma: no cover - model responses must remain available
            logger.exception("Could not configure Qwen3 non-thinking mode.")

    @staticmethod
    def _without_thinking(content: str) -> str:
        """Hide a reasoning block if an older chat template still emits one."""
        return _THINKING_BLOCK.sub("", content).strip()

    def _generation_metrics(self, completion: dict, started_at: float, output_text: str = "") -> dict:
        """Return throughput for the output generated by one LLM request."""
        elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
        usage = completion.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        # Some llama-cpp-python chat-completion versions omit usage. Count the
        # returned text with the loaded model tokenizer so the dashboard still
        # reports a useful throughput value.
        if completion_tokens == 0 and output_text and self._llm is not None:
            try:
                completion_tokens = len(self._llm.tokenize(output_text.encode("utf-8"), add_bos=False))
            except Exception:  # pragma: no cover - metrics must never break a reply
                logger.debug("Could not tokenize generated output for metrics.", exc_info=True)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        total_elapsed = elapsed_seconds
        decode_elapsed = float(usage.get("completion_time", 0) or 0)
        decode_rate = completion_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
        return {
            "completion_tokens": completion_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "prompt_tokens": prompt_tokens,
            "total_elapsed_seconds": round(total_elapsed, 3),
            "tokens_per_second": round(decode_rate or completion_tokens / elapsed_seconds, 2),
        }

    def load(self) -> None:
        """
        Loads the GGUF model into RAM once. llama.cpp automatically KV-caches
        the shared system-prompt prefix across calls that share it, which is
        why the persona prompt should stay static (Tier 1 memory).
        """
        if self._loaded:
            return
        if not _LLAMA_CPP_AVAILABLE:
            logger.warning(
                "llama-cpp-python not installed - LLM engine running in STUB mode. "
                "Install requirements.txt and place a GGUF model at %s to enable real inference.",
                LLM_MODEL_PATH,
            )
            self._loaded = True
            return
        model_path = LLM_MODELS.get(self._active_model_id, {}).get("path", LLM_MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                "No GGUF model found at %s - LLM engine running in STUB mode until a model is placed there.",
                model_path,
            )
            self._loaded = True
            return

        logger.info("Loading brain model from %s ...", model_path)
        kwargs = dict(
            model_path=str(model_path), n_ctx=LLM_CONTEXT_SIZE, n_threads=LLM_THREADS,
            n_threads_batch=LLM_THREADS_BATCH, n_batch=LLM_BATCH,
            n_gpu_layers=LLM_GPU_LAYERS, use_mlock=LLM_MLOCK, verbose=False,
        )
        if LLM_FLASH_ATTN:
            kwargs["flash_attn"] = True
            if LLM_KV_CACHE_TYPE:
                kwargs["type_k"] = LLM_KV_CACHE_TYPE
                kwargs["type_v"] = LLM_KV_CACHE_TYPE
        try:
            self._llm = Llama(**kwargs)
        except TypeError:
            kwargs.pop("flash_attn", None)
            kwargs.pop("type_k", None)
            kwargs.pop("type_v", None)
            self._llm = Llama(**kwargs)
            logger.warning("llama-cpp-python does not support flash attention/KV cache options.")
        if re.match(r"^qwen3(\.\d+)?-", self._active_model_id) and self._llm.chat_handler is None:
            logger.warning("Qwen3 model loaded without explicit thinking disable support.")
        if hasattr(self._llm, "set_cache"):
            try:
                from llama_cpp import LlamaRAMCache
                self._llm.set_cache(LlamaRAMCache(capacity_bytes=LLM_CACHE_MB * 1024 * 1024))
            except (ImportError, TypeError, AttributeError):
                logger.warning("llama-cpp-python cache API unavailable; continuing without RAM cache.")
        self._configure_qwen3_non_thinking_mode()
        self._loaded = True
        logger.info("Brain model loaded (ctx=%s, threads=%s).", LLM_CONTEXT_SIZE, LLM_THREADS)

    # ------------------------------------------------------------------
    # Tool-calling generation - used by the main pipeline in main.py
    # ------------------------------------------------------------------
    async def generate_with_tools(
        self,
        user_text: str,
        tool_schemas: list,
        memory_context: str = "",
        max_tokens: Optional[int] = None,
        history: Optional[list[dict]] = None,
        intent: str = "chat",
    ) -> dict:
        """
        Returns either:
          {"type": "tool_call", "name": ..., "arguments": {...}}
          {"type": "text", "content": "..."}
        """
        self.load()
        max_tokens = max_tokens or (LLM_MAX_TOKENS_TOOL if tool_schemas else LLM_MAX_TOKENS_CHAT)
        messages = self._build_messages(user_text, memory_context, tool_schemas, history, intent)

        if self._llm is None:
            # STUB mode (no model loaded yet) - echo back so the pipeline is testable end-to-end.
            return {"type": "text", "content": f"[stub-brain] You said: {user_text}"}

        started_at = time.perf_counter()
        kwargs = {"messages": messages, "max_tokens": max_tokens,
                  "temperature": LLM_TEMP_TOOLS if tool_schemas else 0.4}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = (
                {"type": "function", "function": {"name": tool_schemas[0]["function"]["name"]}}
                if len(tool_schemas) == 1 else "auto"
            )
        async with self._lock:
            completion = await asyncio.to_thread(self._llm.create_chat_completion, **kwargs)
        logger.info("LLM prompt_tokens=%s", completion.get("usage", {}).get("prompt_tokens", 0))
        choice = completion["choices"][0]["message"]
        metrics = self._generation_metrics(
            completion,
            started_at,
            choice.get("content", "") or json.dumps(choice.get("tool_calls", [])),
        )

        tool_calls = choice.get("tool_calls")
        if tool_calls:
            call = tool_calls[0]["function"]
            try:
                arguments = json.loads(call.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            return {
                "type": "tool_call",
                "name": call["name"],
                "arguments": arguments,
                "metrics": metrics,
            }

        return {
            "type": "text",
            "content": self._without_thinking(choice.get("content", "")),
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Plain generation - used for the second pass (turning a tool result
    # back into spoken confirmation) and by the memory agent.
    # ------------------------------------------------------------------
    async def generate_raw(self, system_prompt: str, user_prompt: str, max_tokens: int = 128) -> str:
        text, _metrics = await self.generate_raw_with_metrics(system_prompt, user_prompt, max_tokens)
        return text

    async def generate_raw_with_metrics(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 128
    ) -> tuple[str, dict]:
        self.load()
        if self._llm is None:
            return "{}", {"completion_tokens": 0, "elapsed_seconds": 0.0, "tokens_per_second": 0.0}
        started_at = time.perf_counter()
        async with self._lock:
            completion = await asyncio.to_thread(
                self._llm.create_chat_completion,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens, temperature=0.2,
            )
        logger.info("LLM prompt_tokens=%s", completion.get("usage", {}).get("prompt_tokens", 0))
        text = self._without_thinking(completion["choices"][0]["message"].get("content", ""))
        return text, self._generation_metrics(completion, started_at, text)

    async def stream_chat(
        self, user_text: str, memory_context: str = "", history: Optional[list[dict]] = None,
        intent: str = "chat",
    ) -> AsyncIterator[dict]:
        """Stream chat deltas while serializing access to the single model."""
        self.load()
        if self._llm is None:
            yield {"type": "text_delta", "text": f"[stub-brain] You said: {user_text}"}
            yield {"type": "done", "metrics": {"completion_tokens": 0, "prompt_tokens": 0, "elapsed_seconds": 0.0, "total_elapsed_seconds": 0.0, "tokens_per_second": 0.0}}
            return
        messages = self._build_messages(user_text, memory_context, [], history, intent)
        values: queue.Queue = queue.Queue()
        started_at = time.perf_counter()

        def produce() -> None:
            try:
                completion = self._llm.create_chat_completion(
                    messages=messages, max_tokens=LLM_MAX_TOKENS_CHAT, temperature=0.4, stream=True
                )
                for item in completion:
                    values.put(item)
            except Exception as exc:
                values.put(exc)
            finally:
                values.put(None)

        async with self._lock:
            worker = threading.Thread(target=produce, daemon=True)
            worker.start()
            pieces = []
            first_token_at = None
            prompt_tokens = 0
            while True:
                item = await asyncio.to_thread(values.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                prompt_tokens = int(item.get("usage", {}).get("prompt_tokens", prompt_tokens))
                delta = item.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                if delta:
                    first_token_at = first_token_at or time.perf_counter()
                    pieces.append(delta)
                    yield {"type": "text_delta", "text": self._without_thinking(delta)}
            output = "".join(pieces)
            elapsed = max(time.perf_counter() - started_at, 0.001)
            completion_tokens = len(self._llm.tokenize(output.encode(), add_bos=False)) if output else 0
            metrics = {
                "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens,
                "elapsed_seconds": round(elapsed, 3), "total_elapsed_seconds": round(elapsed, 3),
                "tokens_per_second": round(completion_tokens / elapsed, 2),
                "ttft_seconds": round((first_token_at or time.perf_counter()) - started_at, 3),
                "decode_tokens_per_second": round(completion_tokens / elapsed, 2),
            }
            logger.info("LLM prompt_tokens=%s", metrics["prompt_tokens"])
            yield {"type": "done", "text": self._without_thinking(output), "metrics": metrics}

    @staticmethod
    def _current_date_context(include_lookup: bool = False) -> str:
        """
        Returns the current date plus a lookup table of the next 14 days'
        dates and weekday names. Small local models (e.g. Qwen3-1.7B) can
        reliably READ "Friday = 2026-08-28" off a list, but cannot reliably
        COMPUTE "today + 4 days" via arithmetic - so give them the answer
        pre-computed instead of asking them to derive it.
        """
        now = datetime.datetime.now().astimezone()
        lines = [now.strftime("Current date: %A, %Y-%m-%d %Z")]
        if not include_lookup:
            return lines[0]
        lines.append("Upcoming dates (use these directly - do not calculate offsets yourself):")
        for offset in range(14):
            day = now + datetime.timedelta(days=offset)
            label = "Today" if offset == 0 else ("Tomorrow" if offset == 1 else day.strftime("%A"))
            lines.append(f"  {label}: {day.strftime('%Y-%m-%d')}")
        return "\n".join(lines)


    @staticmethod
    def _build_system_prompt(memory_context: str = "", tool_schemas: list | None = None) -> str:
        parts = [NOVA_PERSONA_PROMPT]
        if memory_context:
            parts.append(f"Known user context: {memory_context}")
        return "\n\n".join(parts)

    @classmethod
    def _build_messages(cls, user_text, memory_context, tool_schemas, history, intent):
        messages = [{"role": "system", "content": NOVA_PERSONA_PROMPT}]
        for turn in history or []:
            if turn.get("role") in {"user", "assistant"}:
                messages.append({"role": turn["role"], "content": turn.get("content", "")})
        context = cls._current_date_context(intent == "action_calendar")
        if memory_context:
            context += f"\n\nKnown user context: {memory_context}"
        return messages + [{"role": "user", "content": f"[Context]\n{context}\n[/Context]\n\n{user_text}"}]


llm_engine = LLMEngine()