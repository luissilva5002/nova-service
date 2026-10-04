"""Lazy KV residency for serialized conversations sharing one llama context."""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Iterator, Literal

from nova.brain.context_manager import ContextManager
from nova.brain.kv_state import restore_snapshot, save_snapshot, snapshot_nbytes

logger = logging.getLogger("nova.lanes")


@dataclass(frozen=True)
class LaneSpec:
    name: str
    system_prompt: str
    tools: list[dict] | None = None
    gen_reserve: int = 512


class Lane:
    def __init__(
        self,
        key: str,
        kind: Literal["chat", "action"],
        spec: LaneSpec,
        context: ContextManager,
        manager: LaneManager,
    ) -> None:
        self.key = key
        self.kind = kind
        self.spec = spec
        self.context = context
        self._manager = manager

    @property
    def warm(self) -> bool:
        return self.key in self._manager._snapshots

    def matches_llama(self, llama) -> bool:
        n_tokens = int(llama.n_tokens)
        if n_tokens <= 0:
            return False
        ids = self.context.history_ids if self.kind == "chat" else self.context.system_ids
        shared = min(n_tokens, len(ids))
        if shared <= 0:
            return False
        return all(
            int(llama.input_ids[index]) == ids[index]
            for index in range(shared)
        )


class LaneManager:
    def __init__(
        self,
        llama,
        backend,
        n_ctx: int,
        max_snapshot_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._llama = llama
        self._backend = backend
        self._n_ctx = n_ctx
        self._max_snapshot_bytes = max_snapshot_bytes
        self._lanes: dict[str, Lane] = {}
        self._chat_keys: dict[tuple[str, str], str] = {}
        self._snapshots: dict[str, tuple[object, int]] = {}
        self._resident: Lane | None = None
        self._active = False
        self._use_clock = 0
        self._stats = {
            "swaps_in": 0,
            "swaps_out": 0,
            "cold_starts": 0,
            "restore_failures": 0,
            "evictions": 0,
            "warm_prefill_tokens": 0,
        }

    def action_lane(self, spec: LaneSpec) -> Lane:
        key = f"action:{spec.name}"
        lane = self._lanes.get(key)
        if lane is not None:
            if lane.spec != spec:
                raise ValueError(f"Action lane {spec.name!r} already exists with a different spec.")
            return lane
        lane = self._create_lane(key, "action", spec)
        self._lanes[key] = lane
        return lane

    def chat_lane(
        self,
        session_id: str,
        spec: LaneSpec,
        history: list[dict] | None = None,
    ) -> Lane:
        identity = (spec.name, session_id)
        key = self._chat_keys.get(identity)
        if key is not None:
            lane = self._lanes[key]
            if lane.spec != spec:
                raise ValueError(f"Chat lane {identity!r} already exists with a different spec.")
            return lane
        key = f"chat:{len(spec.name)}:{spec.name}:{len(session_id)}:{session_id}"
        lane = self._create_lane(key, "chat", spec)
        if history is not None:
            lane.context.load_history(history)
        self._lanes[key] = lane
        self._chat_keys[identity] = key
        return lane

    def forget_chat(self, session_id: str) -> None:
        identities = [identity for identity in self._chat_keys if identity[1] == session_id]
        for identity in identities:
            key = self._chat_keys.pop(identity)
            lane = self._lanes.pop(key)
            self._drop_snapshot(key)
            if self._resident is lane:
                self._llama.reset()
                self._resident = None
        logger.debug("Forgot chat session %s (%d lanes)", session_id, len(identities))

    def warm(self, lane: Lane) -> int:
        self._require_owned(lane)
        if lane.kind != "action":
            raise ValueError("Only action lanes can be explicitly warmed.")
        if lane.warm:
            self._touch_snapshot(lane.key)
            return 0
        if self._active:
            raise RuntimeError("Cannot warm a lane while another lane context is active.")
        return self._warm_action(lane)

    @contextlib.contextmanager
    def use(self, lane: Lane) -> Iterator[Lane]:
        self._enter_guard()
        try:
            self._require_owned(lane)
            if not (self._resident is lane and lane.matches_llama(self._llama)):
                self._park_resident()
                self._resident = None
                self._load_lane(lane)
                self._resident = lane
            self._touch_snapshot(lane.key)
            yield lane
        finally:
            self._active = False

    @contextlib.contextmanager
    def borrow(self) -> Iterator[None]:
        self._enter_guard()
        try:
            self._park_resident()
            self._resident = None
            yield
        finally:
            self._active = False

    def stats(self) -> dict:
        snapshot_bytes = sum(snapshot_nbytes(snapshot) for snapshot, _ in self._snapshots.values())
        return {
            **self._stats,
            "snapshot_bytes": snapshot_bytes,
            "snapshots": len(self._snapshots),
            "resident": self._resident.key if self._resident is not None else None,
        }

    def _create_lane(
        self, key: str, kind: Literal["chat", "action"], spec: LaneSpec
    ) -> Lane:
        context = ContextManager(
            self._backend,
            n_ctx=self._n_ctx,
            gen_reserve=spec.gen_reserve,
        )
        context.start_session(spec.system_prompt, spec.tools)
        return Lane(key, kind, spec, context, self)

    def _require_owned(self, lane: Lane) -> None:
        if self._lanes.get(lane.key) is not lane:
            raise ValueError("Lane does not belong to this manager.")

    def _enter_guard(self) -> None:
        if self._active:
            raise RuntimeError("nested lane use")
        self._active = True

    def _park_resident(self) -> None:
        lane = self._resident
        if lane is None or int(self._llama.n_tokens) == 0:
            return
        if not lane.matches_llama(self._llama):
            logger.debug("Skipping snapshot for non-matching resident lane %s", lane.key)
            return
        if lane.kind == "action" and lane.warm:
            return

        snapshot = save_snapshot(self._llama, meta={"lane": lane.key})
        if self._store_snapshot(lane, snapshot):
            self._stats["swaps_out"] += 1

    def _load_lane(self, lane: Lane) -> None:
        stored = self._snapshots.get(lane.key)
        if stored is not None:
            snapshot = stored[0]
            try:
                restore_snapshot(self._llama, snapshot)
            except Exception:
                self._llama.reset()
                self._drop_snapshot(lane.key)
                self._stats["restore_failures"] += 1
                logger.debug("Snapshot restore failed for %s; starting cold", lane.key, exc_info=True)
                if lane.kind == "action":
                    self._warm_action(lane)
                else:
                    self._stats["cold_starts"] += 1
                return
            self._stats["swaps_in"] += 1
            self._touch_snapshot(lane.key)
            return

        if lane.kind == "action":
            self._warm_action(lane)
        else:
            self._llama.reset()
            self._stats["cold_starts"] += 1
            logger.debug("Cold-started chat lane %s", lane.key)

    def _warm_action(self, lane: Lane) -> int:
        self._park_resident()
        self._llama.reset()
        system_ids = lane.context.system_ids
        if system_ids:
            self._llama.eval(system_ids)
        prefilled = len(system_ids)
        self._stats["cold_starts"] += 1
        self._stats["warm_prefill_tokens"] += prefilled
        snapshot = save_snapshot(self._llama, meta={"lane": lane.key})
        self._store_snapshot(lane, snapshot)
        self._resident = lane
        logger.debug("Warmed action lane %s with %d tokens", lane.key, prefilled)
        return prefilled

    def _store_snapshot(self, lane: Lane, snapshot) -> bool:
        size = snapshot_nbytes(snapshot)
        self._drop_snapshot(lane.key)
        if size > self._max_snapshot_bytes:
            self._stats["evictions"] += 1
            logger.debug(
                "Did not store oversized snapshot for %s (%d bytes > %d)",
                lane.key,
                size,
                self._max_snapshot_bytes,
            )
            return False

        self._use_clock += 1
        self._snapshots[lane.key] = (snapshot, self._use_clock)
        while self._snapshot_bytes() > self._max_snapshot_bytes:
            candidates = [
                (key, entry)
                for key, entry in self._snapshots.items()
                if key != lane.key
            ]
            if not candidates:
                self._drop_snapshot(lane.key)
                self._stats["evictions"] += 1
                return False
            candidates.sort(
                key=lambda item: (
                    self._lanes[item[0]].kind != "chat",
                    item[1][1],
                )
            )
            evicted_key = candidates[0][0]
            self._drop_snapshot(evicted_key)
            self._stats["evictions"] += 1
            logger.debug("Evicted snapshot for %s", evicted_key)
        return lane.key in self._snapshots

    def _snapshot_bytes(self) -> int:
        return sum(snapshot_nbytes(snapshot) for snapshot, _ in self._snapshots.values())

    def _drop_snapshot(self, key: str) -> None:
        self._snapshots.pop(key, None)

    def _touch_snapshot(self, key: str) -> None:
        stored = self._snapshots.get(key)
        if stored is not None:
            self._use_clock += 1
            self._snapshots[key] = (stored[0], self._use_clock)
