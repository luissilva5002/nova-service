"""Synchronous chat-turn orchestration over per-session KV-cache lanes."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterator

from nova.brain.context_manager import ContextOverflow, LlamaBackend
from nova.brain.generation import (
    GenerationParams,
    GenerationResult,
    TokenGenerator,
    stop_ids_for,
)
from nova.brain.lanes import Lane, LaneManager, LaneSpec

logger = logging.getLogger("nova.lane_engine")


class LaneOverflow(Exception):
    """A chat turn cannot fit in its lane, even after rebuilding its history."""


@dataclass
class ChatTurnResult:
    text: str
    ids: list[int]
    finish_reason: str
    committed: bool
    metrics: dict


class LaneEngine:
    """Run serialized chat turns while preserving each session's token history."""

    def __init__(
        self,
        llama,
        n_ctx: int,
        system_prompt: str,
        gen_reserve: int = 160,
        load_fraction: float = 0.5,
        compact_fraction: float = 0.35,
        max_snapshot_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._llama = llama
        self._backend = LlamaBackend(llama)
        self._manager = LaneManager(
            llama,
            self._backend,
            n_ctx=n_ctx,
            max_snapshot_bytes=max_snapshot_bytes,
        )
        self._spec = LaneSpec(
            name="chat",
            system_prompt=system_prompt,
            tools=None,
            gen_reserve=gen_reserve,
        )
        self._load_fraction = load_fraction
        self._compact_fraction = compact_fraction
        self._lanes: dict[str, Lane] = {}
        self._stats = {
            "turns": 0,
            "compactions": 0,
            "overflows": 0,
            "external_turns": 0,
        }

    def chat_stream(
        self,
        session_id: str,
        user_text: str,
        context_block: str | None,
        history_messages: list[dict] | None,
        params: GenerationParams,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str | ChatTurnResult]:
        started_at = time.monotonic()
        lane = self._get_or_create_lane(session_id, history_messages)

        if lane.context.needs_compaction():
            lane = self._rebuild_lane(session_id, history_messages)

        retried_after_overflow = False
        while True:
            turn_overflow: ContextOverflow | None = None
            chat_result: ChatTurnResult | None = None
            with self._manager.use(lane):
                turn_open = False
                committed = False
                token_stream = None
                try:
                    try:
                        prompt = lane.context.begin_turn(
                            user_text,
                            context_block,
                            context_position="before",
                        )
                    except ContextOverflow as exc:
                        turn_overflow = exc

                    if turn_overflow is None:
                        turn_open = True
                        stop_ids = stop_ids_for(lane.context, self._llama)
                        token_stream = TokenGenerator(self._llama, stop_ids).stream(
                            prompt,
                            params,
                            should_stop=should_stop,
                        )
                        generation_started = time.monotonic()
                        first_delta_at: float | None = None
                        generation_result: GenerationResult | None = None
                        for item in token_stream:
                            if isinstance(item, str):
                                if first_delta_at is None:
                                    first_delta_at = time.monotonic()
                                yield item
                            else:
                                generation_result = item

                        if generation_result is None:
                            raise RuntimeError(
                                "Token generation ended without a GenerationResult."
                            )

                        if generation_result.finish_reason in ("stop", "length"):
                            lane.context.commit_turn(generation_result.ids)
                            committed = True
                            turn_open = False
                        else:
                            lane.context.abort_turn()
                            turn_open = False

                        completed_at = time.monotonic()
                        elapsed = max(0.0, completed_at - generation_started)
                        ttft = (
                            max(0.0, first_delta_at - generation_started)
                            if first_delta_at is not None
                            else elapsed
                        )
                        completion_tokens = len(generation_result.ids)
                        if (
                            generation_result.finish_reason == "stop"
                            and generation_result.ids
                            and generation_result.ids[-1] in stop_ids
                        ):
                            completion_tokens -= 1
                        rate = completion_tokens / elapsed if elapsed > 0 else 0.0
                        result = ChatTurnResult(
                            text=generation_result.text,
                            ids=list(generation_result.ids),
                            finish_reason=generation_result.finish_reason,
                            committed=committed,
                            metrics={
                                "completion_tokens": completion_tokens,
                                "prompt_tokens": generation_result.prompt_tokens,
                                "elapsed_seconds": elapsed,
                                "total_elapsed_seconds": max(
                                    0.0, completed_at - started_at
                                ),
                                "tokens_per_second": rate,
                                "decode_tokens_per_second": rate,
                                "ttft_seconds": ttft,
                                "cache_hit_tokens": generation_result.reused_tokens,
                                "evaluated_tokens": generation_result.evaluated_tokens,
                                "lane": lane.key,
                            },
                        )
                        self._stats["turns"] += 1
                        chat_result = result

                finally:
                    if token_stream is not None:
                        token_stream.close()
                    if turn_open and not committed:
                        lane.context.abort_turn()

            if chat_result is not None:
                yield chat_result
                return

            if turn_overflow is not None:
                if retried_after_overflow:
                    self._stats["overflows"] += 1
                    raise LaneOverflow(
                        "The chat turn exceeds the lane capacity after a history rebuild."
                    ) from turn_overflow
                lane = self._rebuild_lane(session_id, history_messages)
                retried_after_overflow = True

    def chat(
        self,
        session_id: str,
        user_text: str,
        context_block: str | None,
        history_messages: list[dict] | None,
        params: GenerationParams,
        should_stop: Callable[[], bool] | None = None,
    ) -> ChatTurnResult:
        result: ChatTurnResult | None = None
        for item in self.chat_stream(
            session_id,
            user_text,
            context_block,
            history_messages,
            params,
            should_stop,
        ):
            if isinstance(item, ChatTurnResult):
                result = item
        if result is None:
            raise RuntimeError("Chat stream ended without a ChatTurnResult.")
        return result

    def record_external_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        lane = self._lanes.get(session_id)
        if lane is None:
            return
        try:
            lane.context.append_external_turn(user_text, assistant_text)
        except (ContextOverflow, ValueError) as exc:
            self.forget(session_id)
            if isinstance(exc, ContextOverflow):
                self._stats["overflows"] += 1
            raise
        self._stats["external_turns"] += 1

    def forget(self, session_id: str) -> None:
        self._manager.forget_chat(session_id)
        self._lanes.pop(session_id, None)

    def has_lane(self, session_id: str) -> bool:
        return session_id in self._lanes

    def borrow(self):
        return self._manager.borrow()

    def stats(self) -> dict:
        return {**self._manager.stats(), **self._stats}

    def _get_or_create_lane(
        self,
        session_id: str,
        history_messages: list[dict] | None,
    ) -> Lane:
        lane = self._lanes.get(session_id)
        if lane is not None:
            return lane

        lane = self._manager.chat_lane(session_id, self._spec)
        self._lanes[session_id] = lane
        if history_messages:
            try:
                lane.context.load_history(
                    history_messages,
                    max_fraction=self._load_fraction,
                )
            except ContextOverflow as exc:
                self.forget(session_id)
                self._stats["overflows"] += 1
                raise LaneOverflow("The supplied chat history exceeds lane capacity.") from exc
        return lane

    def _rebuild_lane(
        self,
        session_id: str,
        history_messages: list[dict] | None,
    ) -> Lane:
        self._manager.forget_chat(session_id)
        self._lanes.pop(session_id, None)
        lane = self._manager.chat_lane(session_id, self._spec)
        self._lanes[session_id] = lane
        if history_messages:
            try:
                lane.context.load_history(
                    history_messages,
                    max_fraction=self._compact_fraction,
                )
            except ContextOverflow as exc:
                self.forget(session_id)
                self._stats["overflows"] += 1
                raise LaneOverflow("The supplied chat history cannot be compacted.") from exc
        else:
            logger.warning(
                "Rebuilding chat lane %s without history because no history was supplied",
                session_id,
            )
        self._stats["compactions"] += 1
        return lane
