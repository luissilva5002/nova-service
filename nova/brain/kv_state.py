"""Compact llama.cpp context snapshots without the large logits scores array."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_FINGERPRINTS: dict[str, tuple[int, str]] = {}
_FINGERPRINT_SAMPLE_BYTES = 1024 * 1024
_RESTORE_COMPATIBILITY_KEYS = (
    "n_ctx",
    "n_vocab",
    "llama_cpp_version",
    "model_fingerprint",
    "model_size",
    "flash_attn",
    "type_k",
    "type_v",
)


@dataclass
class KvSnapshot:
    state: bytes
    size: int
    input_ids: np.ndarray
    n_tokens: int
    seed: int | None
    meta: dict[str, Any]


def _state_api():
    import llama_cpp

    new_names = (
        "llama_state_get_size",
        "llama_state_get_data",
        "llama_state_set_data",
    )
    if all(callable(getattr(llama_cpp, name, None)) for name in new_names):
        get_size = llama_cpp.llama_state_get_size

        def copy_out(ctx, buffer, size):
            return int(llama_cpp.llama_state_get_data(ctx, buffer, size))

        def copy_in(ctx, buffer, size):
            return int(llama_cpp.llama_state_set_data(ctx, buffer, size))

    else:
        old_names = (
            "llama_get_state_size",
            "llama_copy_state_data",
            "llama_set_state_data",
        )
        if not all(callable(getattr(llama_cpp, name, None)) for name in old_names):
            raise RuntimeError("llama.cpp context state functions are unavailable.")
        get_size = llama_cpp.llama_get_state_size

        def copy_out(ctx, buffer, size):
            return int(llama_cpp.llama_copy_state_data(ctx, buffer))

        def copy_in(ctx, buffer, size):
            return int(llama_cpp.llama_set_state_data(ctx, buffer))

    return get_size, copy_out, copy_in


def _llama_cpp_version() -> str:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        try:
            import llama_cpp
        except ImportError:
            return "unknown"
        return str(getattr(llama_cpp, "__version__", "unknown"))


def _dimensions(llama) -> tuple[int, int]:
    return int(llama.n_ctx()), int(llama.n_vocab())


def _model_fingerprint(model_path: Path, model_size: int) -> str:
    cache_key = str(model_path.resolve())
    cached = _MODEL_FINGERPRINTS.get(cache_key)
    if cached is not None and cached[0] == model_size:
        return cached[1]

    sample_size = _FINGERPRINT_SAMPLE_BYTES
    middle_offset = max(0, (model_size - sample_size) // 2)
    last_offset = max(0, model_size - sample_size)
    digest = hashlib.sha256()
    digest.update(str(model_size).encode("ascii"))
    with model_path.open("rb") as model_file:
        for offset in (0, middle_offset, last_offset):
            model_file.seek(offset)
            digest.update(model_file.read(sample_size))
    fingerprint = digest.hexdigest()
    _MODEL_FINGERPRINTS[cache_key] = (model_size, fingerprint)
    return fingerprint


def _stored_constructor_value(llama, name: str):
    values = llama.__dict__
    if name in values:
        return values[name]
    for kwargs_name in ("constructor_kwargs", "_constructor_kwargs", "kwargs", "_kwargs"):
        kwargs = values.get(kwargs_name)
        if isinstance(kwargs, dict) and name in kwargs:
            return kwargs[name]
    return None


def _context_option(llama, name: str):
    context_params = getattr(llama, "context_params", None)
    if context_params is not None and hasattr(context_params, name):
        return getattr(context_params, name)
    return _stored_constructor_value(llama, name)


def live_meta(llama, extra: dict | None = None) -> dict:
    """Return model and context identity metadata for a live llama instance."""
    model_path = Path(llama.model_path)
    model_size = model_path.stat().st_size
    n_ctx, n_vocab = _dimensions(llama)
    metadata = dict(extra or {})
    metadata.update(
        {
            "n_ctx": n_ctx,
            "n_vocab": n_vocab,
            "llama_cpp_version": _llama_cpp_version(),
            "model_name": model_path.name,
            "model_size": model_size,
            "model_fingerprint": _model_fingerprint(model_path, model_size),
            "flash_attn": _context_option(llama, "flash_attn"),
            "type_k": _context_option(llama, "type_k"),
            "type_v": _context_option(llama, "type_v"),
        }
    )
    return metadata


def save_snapshot(llama, meta: dict | None = None) -> KvSnapshot:
    """Capture serialized context state and only the active prompt token IDs."""
    get_size, copy_out, _ = _state_api()
    ctx = llama._ctx.ctx
    buffer_size = int(get_size(ctx))
    if buffer_size <= 0:
        raise RuntimeError(f"llama.cpp returned an invalid state size: {buffer_size}")

    buffer = (ctypes.c_uint8 * buffer_size)()
    copied_size = copy_out(ctx, buffer, buffer_size)
    if copied_size <= 0 or copied_size > buffer_size:
        raise RuntimeError(
            f"llama.cpp copied an invalid state byte count: {copied_size} "
            f"(buffer size {buffer_size})"
        )

    n_tokens = int(llama.n_tokens)
    if n_tokens < 0 or n_tokens > len(llama.input_ids):
        raise RuntimeError(f"llama.cpp returned an invalid token count: {n_tokens}")

    snapshot_meta = live_meta(llama, meta)
    return KvSnapshot(
        state=ctypes.string_at(ctypes.addressof(buffer), copied_size),
        size=copied_size,
        input_ids=np.asarray(llama.input_ids[:n_tokens]).copy(),
        n_tokens=n_tokens,
        seed=llama._seed,
        meta=snapshot_meta,
    )


def restore_snapshot(llama, snap: KvSnapshot) -> None:
    """Restore a snapshot only into a context with matching model identity."""
    current_meta = live_meta(llama)
    differences = {
        key: (snap.meta[key], current_meta.get(key))
        for key in _RESTORE_COMPATIBILITY_KEYS
        if key in snap.meta and snap.meta[key] != current_meta.get(key)
    }
    if differences:
        details = ", ".join(
            f"{key}: snapshot={snapshot_value!r}, live={live_value!r}"
            for key, (snapshot_value, live_value) in differences.items()
        )
        raise ValueError(
            f"Snapshot metadata differs from the live llama context: {details}"
        )
    if not isinstance(snap.state, bytes):
        raise ValueError("Snapshot state must be bytes.")
    if snap.size != len(snap.state):
        raise ValueError(
            f"Snapshot size metadata ({snap.size}) does not match state bytes "
            f"({len(snap.state)})."
        )
    if snap.size <= 0:
        raise ValueError("Snapshot state must contain at least one byte.")
    if not isinstance(snap.input_ids, np.ndarray) or (
        snap.input_ids.ndim != 1
        or not np.issubdtype(snap.input_ids.dtype, np.integer)
        or snap.n_tokens != len(snap.input_ids)
    ):
        raise ValueError(
            "Snapshot input IDs must be a one-dimensional integer array whose "
            f"length matches n_tokens ({snap.n_tokens})."
        )
    if snap.n_tokens < 0:
        raise ValueError(f"Snapshot token count cannot be negative: {snap.n_tokens}.")
    if snap.n_tokens > len(llama.input_ids):
        raise ValueError("Snapshot token IDs exceed the live context capacity.")

    _, _, copy_in = _state_api()
    buffer = (ctypes.c_uint8 * snap.size).from_buffer_copy(snap.state)
    try:
        restored_size = copy_in(llama._ctx.ctx, buffer, snap.size)
        if restored_size != snap.size:
            raise RuntimeError(
                f"llama.cpp restored {restored_size} bytes; expected {snap.size}."
            )
    except Exception:
        llama.reset()
        raise

    llama.input_ids[: snap.n_tokens] = snap.input_ids
    llama.n_tokens = snap.n_tokens
    llama._seed = snap.seed
    if hasattr(llama, "_requires_eval"):
        llama._requires_eval = True


def snapshot_to_file(snap: KvSnapshot, path: str | os.PathLike[str]) -> None:
    """Write a pickle-free NPZ snapshot using an atomic same-directory replace."""
    if snap.size != len(snap.state):
        raise ValueError(
            f"Snapshot size metadata ({snap.size}) does not match state bytes "
            f"({len(snap.state)})."
        )
    if snap.n_tokens != len(snap.input_ids):
        raise ValueError(
            f"Snapshot token count ({snap.n_tokens}) does not match input IDs "
            f"({len(snap.input_ids)})."
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            np.savez(
                temp_file,
                state=np.frombuffer(snap.state, dtype=np.uint8),
                input_ids=np.asarray(snap.input_ids),
                meta=np.asarray(json.dumps(snap.meta, ensure_ascii=False)),
                seed=np.asarray(json.dumps(snap.seed)),
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def snapshot_from_file(
    path: str | os.PathLike[str], expected_meta: dict | None = None
) -> KvSnapshot | None:
    """Load a valid NPZ snapshot, returning None for unreadable or mismatched files."""
    try:
        with open(path, "rb") as snapshot_file:
            with np.load(snapshot_file, allow_pickle=False) as archive:
                if not {"state", "input_ids", "meta", "seed"}.issubset(archive.files):
                    return None
                state_array = archive["state"]
                input_ids = archive["input_ids"]
                if state_array.dtype != np.dtype(np.uint8):
                    return None
                if not np.issubdtype(input_ids.dtype, np.integer):
                    return None
                meta_value = archive["meta"]
                seed_value = archive["seed"]
                if meta_value.ndim != 0 or seed_value.ndim != 0:
                    return None
                stored_meta = json.loads(str(meta_value.item()))
                seed = json.loads(str(seed_value.item()))
                if not isinstance(stored_meta, dict):
                    return None
                if expected_meta is not None and any(
                    stored_meta.get(key, object()) != value
                    for key, value in expected_meta.items()
                ):
                    return None

                stored_ids = np.asarray(input_ids).copy()
                state = state_array.tobytes()
                return KvSnapshot(
                    state=state,
                    size=len(state),
                    input_ids=stored_ids,
                    n_tokens=len(stored_ids),
                    seed=seed,
                    meta=stored_meta,
                )
    except Exception:
        return None


def load_snapshot_for(
    llama, path: str | os.PathLike[str], extra_meta: dict | None = None
) -> KvSnapshot | None:
    """Load a snapshot only when its stored identity matches this llama instance."""
    try:
        return snapshot_from_file(path, expected_meta=live_meta(llama, extra_meta))
    except Exception:
        return None


def snapshot_nbytes(snap: KvSnapshot) -> int:
    """Return serialized context bytes plus active token-ID bytes."""
    return len(snap.state) + int(snap.input_ids.nbytes)
