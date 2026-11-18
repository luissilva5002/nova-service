"""In-place edits of a live llama.cpp KV cache (KV-C-lite).

Chat turns are evaluated with a per-turn ``[Context]`` block in the user
message, but the history stores the *clean* user message (see
``ContextManager.commit_turn``). The KV cache therefore still holds the block
after the turn, diverges from the history at that user message, and the next
turn re-evaluates the whole previous exchange. ``strip_context`` removes the
block's tokens from the cache and slides the later tokens down, so the cache
matches the history again.

Everything here works on one sequence (seq_id 0). In llama-cpp-python 0.3.36
``kv_cache_seq_shift`` wraps ``llama_memory_seq_add`` and rejects seq_id -1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


class KvEditRefused(Exception):
    """The edit was not started or was refused before the cache changed."""


class KvEditCorrupted(Exception):
    """The cache may be half-edited; the caller must reset the llama context."""


@dataclass
class StripResult:
    status: str  # "stripped" | "skipped" | "failed"
    reason: str = ""
    removed_tokens: int = 0
    seconds: float = 0.0


def _shift_function(ctx):
    for name in ("kv_cache_seq_shift", "kv_cache_seq_add"):
        function = getattr(ctx, name, None)
        if callable(function):
            return function
    return None


def kv_remove_span(llama, start: int, end: int) -> None:
    """Delete tokens [start, end) from seq 0; tokens at and after ``end`` slide down.

    Keeps ``llama.input_ids`` and ``llama.n_tokens`` in step with the cache.
    Raises KvEditRefused if nothing changed, KvEditCorrupted if it may have.
    """
    n_tokens = int(llama.n_tokens)
    if not 0 <= start < end <= n_tokens:
        raise KvEditRefused(f"span [{start}, {end}) is outside the {n_tokens} cached tokens")
    ctx = getattr(llama, "_ctx", None)
    remove = getattr(ctx, "kv_cache_seq_rm", None)
    shift = _shift_function(ctx)
    if not callable(remove) or (shift is None and end < n_tokens):
        raise KvEditRefused("the KV sequence API is unavailable in this llama-cpp-python")

    removed = end - start
    if remove(0, start, end) is False:
        raise KvEditRefused("kv_cache_seq_rm refused the range")
    try:
        if end < n_tokens:
            if shift(0, end, n_tokens, -removed) is False:
                raise KvEditCorrupted("kv_cache_seq_shift returned False after the removal")
        ids = llama.input_ids
        ids[start : n_tokens - removed] = ids[end:n_tokens].copy()
        llama.n_tokens = n_tokens - removed
    except KvEditCorrupted:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure after the removal leaves the cache unusable
        raise KvEditCorrupted(f"{type(exc).__name__}: {exc}") from exc


def align_context_span(shown: list[int], clean: list[int]) -> tuple[int, int] | None:
    """Locate the tokens ``shown`` has beyond ``clean``.

    Returns (offset inside the segment, number of extra tokens) only when
    ``shown`` is exactly ``clean`` with one contiguous block inserted. Returns
    None when there is no block, or when tokenization merged tokens across the
    block's boundary so that the clean tokens are not a prefix + suffix of shown.
    """
    if len(shown) <= len(clean):
        return None
    prefix = 0
    while prefix < len(clean) and shown[prefix] == clean[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(clean) - prefix and shown[-1 - suffix] == clean[-1 - suffix]:
        suffix += 1
    if prefix + suffix != len(clean):
        return None
    return prefix, len(shown) - len(clean)


def strip_context(
    llama,
    history_prefix: list[int],
    shown: list[int],
    clean: list[int],
) -> StripResult:
    """Remove the retrieved-context tokens of the user segment just committed.

    ``history_prefix`` is the history before that user segment (its length is
    the segment's start position). The cache must still hold history_prefix +
    shown (+ the reply), otherwise nothing is touched. On success the cache
    holds history_prefix + clean (+ the reply).
    """
    started = time.perf_counter()

    def done(status: str, reason: str = "", removed: int = 0) -> StripResult:
        return StripResult(status, reason, removed, time.perf_counter() - started)

    span = align_context_span(shown, clean)
    if span is None:
        reason = "no_context" if len(shown) <= len(clean) else "boundary_tokens_differ"
        return done("skipped", reason)
    offset, removed = span

    start = len(history_prefix)
    n_tokens = int(llama.n_tokens)
    if n_tokens < start + len(shown):
        return done("skipped", "kv_shorter_than_segment")
    cached = np.asarray(llama.input_ids[: start + len(shown)])
    expected = np.concatenate(
        (np.asarray(history_prefix, dtype=cached.dtype), np.asarray(shown, dtype=cached.dtype))
    )
    if not np.array_equal(cached, expected):
        return done("skipped", "kv_differs_from_history")

    try:
        kv_remove_span(llama, start + offset, start + offset + removed)
    except KvEditRefused as exc:
        return done("skipped", f"refused: {exc}")
    except KvEditCorrupted as exc:
        llama.reset()
        return done("failed", f"corrupted: {exc}")

    after = np.asarray(llama.input_ids[start : start + len(clean)])
    if not np.array_equal(after, np.asarray(clean, dtype=after.dtype)):
        llama.reset()
        return done("failed", "post_check_mismatch")
    return done("stripped", "", removed)
