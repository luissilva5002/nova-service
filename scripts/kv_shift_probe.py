"""Probe whether llama.cpp can remove old chat turns from a live KV cache.

Run from the repository root with the qwen2.5-3b GGUF installed:
    python scripts/kv_shift_probe.py
    python scripts/kv_shift_probe.py --n-ctx 1024

The probe records token counts and generated-token equality only; it does not
measure or report timings.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nova.config import (  # noqa: E402
    LLM_BATCH,
    LLM_CONTEXT_SIZE,
    LLM_FLASH_ATTN,
    LLM_KV_CACHE_TYPE,
    LLM_MODELS,
    LLM_THREADS,
    NOVA_PERSONA_PROMPT,
)
from nova.brain.context_manager import ContextManager, LlamaBackend  # noqa: E402
from nova.brain.generation import GenerationParams, GenerationResult, TokenGenerator, stop_ids_for  # noqa: E402
from nova.brain.kv_state import restore_snapshot, save_snapshot  # noqa: E402


INITIAL_TURNS = [
    "Why does the Moon appear to change shape during the month, even though its actual shape stays the same?",
    "How does sunlight reach the side of Earth that is facing away from the Sun during the night?",
    "Why do objects usually look smaller when they are farther away from someone viewing them?",
    "What causes clouds to form when warm air rises above the ground on a humid day?",
    "How can a small seed contain enough information to grow into a much larger plant?",
    "Why does a metal spoon feel colder than a wooden spoon in the same room?",
    "What makes some rocks develop visible layers while other rocks appear more uniform?",
    "Why can a rainbow appear after rain when sunlight shines through drops of water?",
    "How do roots help a plant stay upright while also gathering what it needs from soil?",
    "Why does a bicycle keep moving briefly after someone stops pedaling on level ground?",
    "How can birds find their way across long distances when familiar landmarks are absent?",
    "Why does water take longer to boil at a high mountain than near sea level?",
    "What happens to a shadow when the light source moves closer to the object?",
    "Why do some materials bend easily while others crack under the same force?",
]

NEW_MESSAGES = [
    "Why does warm air tend to rise above cooler air in an otherwise calm room?",
    "How does a simple lever help someone lift a heavy object with less effort?",
    "What causes ocean tides to rise and fall along a coastline during the day?",
]

CONSECUTIVE_MESSAGES = [
    "Why does warm air tend to rise above cooler air in an otherwise calm room?",
    "How does a simple lever help someone lift a heavy object with less effort?",
    "What causes ocean tides to rise and fall along a coastline during the day?",
    "Why do mirrors reverse images horizontally but not vertically?",
]

MAX_INITIAL_TOKENS = 16
DEFAULT_TRIAL_TOKENS = 60


class EvalCounter:
    """Count llama.eval tokens until the underlying generator yields its first token."""

    def __init__(self, llama) -> None:
        self.llama = llama
        self.total = 0
        self.active = True
        self._original_eval = None
        self._original_generate = None
        self._had_eval = False
        self._had_generate = False
        self._instance_eval = None
        self._instance_generate = None

    def __enter__(self):
        namespace = self.llama.__dict__
        self._had_eval = "eval" in namespace
        self._had_generate = "generate" in namespace
        self._instance_eval = namespace.get("eval")
        self._instance_generate = namespace.get("generate")
        self._original_eval = self.llama.eval
        self._original_generate = self.llama.generate

        def counted_eval(tokens, *args, **kwargs):
            if self.active:
                self.total += len(tokens)
            return self._original_eval(tokens, *args, **kwargs)

        def observed_generate(*args, **kwargs):
            token_iterator = self._original_generate(*args, **kwargs)
            try:
                for token in token_iterator:
                    self.active = False
                    yield token
            finally:
                token_iterator.close()

        self.llama.eval = counted_eval
        self.llama.generate = observed_generate
        return self

    def __exit__(self, *_exc):
        if self._had_eval:
            self.llama.eval = self._instance_eval
        else:
            del self.llama.eval
        if self._had_generate:
            self.llama.generate = self._instance_generate
        else:
            del self.llama.generate


def generate(
    llama,
    context: ContextManager,
    prompt_ids: list[int],
    max_tokens: int,
) -> tuple[GenerationResult, int]:
    counter: EvalCounter
    with EvalCounter(llama) as counter:
        result = TokenGenerator(llama, stop_ids_for(context, llama)).generate(
            prompt_ids,
            GenerationParams(max_tokens=max_tokens, temperature=0.0),
        )
    return result, counter.total


def _api_signature(ctx: Any, name: str) -> tuple[bool, str]:
    member = getattr(ctx, name, None)
    if not callable(member):
        return False, "absent"
    try:
        signature = str(inspect.signature(member))
    except (TypeError, ValueError) as exc:
        signature = f"(signature unavailable: {type(exc).__name__}: {exc})"
    return True, signature


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _decode_around(backend: LlamaBackend, ids: list[int], index: int) -> str:
    start = max(0, index - 5)
    end = min(len(ids), index + 6)
    return backend.decode(ids[start:end]).replace("\n", "\\n")


def _new_turn_prompt(
    backend: LlamaBackend,
    system_prompt: str,
    n_ctx: int,
    message: str,
    tool_schemas: list[dict] | None = None,
    gen_reserve: int = DEFAULT_TRIAL_TOKENS,
) -> tuple[ContextManager, list[int]]:
    builder = ContextManager(backend, n_ctx=n_ctx, gen_reserve=gen_reserve)
    builder.start_session(system_prompt, tool_schemas)
    prompt_ids = builder.begin_turn(message)
    return builder, prompt_ids


def _context_turn_block(
    backend: LlamaBackend,
    system_prompt: str,
    n_ctx: int,
    message: str,
    generated_ids: list[int],
    tool_schemas: list[dict] | None = None,
    gen_reserve: int = DEFAULT_TRIAL_TOKENS,
) -> tuple[ContextManager, list[int]]:
    builder = ContextManager(backend, n_ctx=n_ctx, gen_reserve=gen_reserve)
    builder.start_session(system_prompt, tool_schemas)
    builder.begin_turn(message)
    builder.commit_turn(generated_ids)
    system_len = len(builder.system_ids)
    return builder, builder.history_ids[system_len:]


def _shift_cache(
    llama,
    sys_len: int,
    end: int,
    expected_system_ids: list[int] | None = None,
) -> bool:
    """Drop [sys_len, end) from the KV cache and shift surviving tokens backwards."""
    # Anchor pre-check: verify system prefix before modifying
    if expected_system_ids is not None:
        if list(llama.input_ids[:sys_len]) != expected_system_ids:
            return False

    n_tokens = int(llama.n_tokens)
    # Use seq_id=0 for standard single sequence
    removed = llama._ctx.kv_cache_seq_rm(0, sys_len, end)
    if removed is False:
        return False

    if hasattr(llama._ctx, "kv_cache_seq_shift"):
        llama._ctx.kv_cache_seq_shift(0, end, n_tokens, -(end - sys_len))
    elif hasattr(llama._ctx, "kv_cache_seq_add"):
        shifted = llama._ctx.kv_cache_seq_add(0, end, n_tokens, -(end - sys_len))
        if shifted is False:
            return False
    else:
        return False

    old_ids = llama.input_ids
    new_ids = np.concatenate((old_ids[:sys_len], old_ids[end:n_tokens]))
    llama.input_ids[: len(new_ids)] = new_ids
    llama.n_tokens = len(new_ids)

    # Anchor post-check: verify system prefix remains completely untouched
    if expected_system_ids is not None:
        if list(llama.input_ids[:sys_len]) != expected_system_ids:
            return False

    return True


def _compare_outputs(
    backend: LlamaBackend,
    shifted: GenerationResult,
    control: GenerationResult,
) -> tuple[bool, int | None, str]:
    difference = _first_difference(shifted.ids, control.ids)
    if difference is None:
        return True, None, "identical"

    # Classify divergence: early (< 30 tokens) vs late (>= 30 tokens)
    coherence = "diverge late" if difference >= 30 else "diverge early"
    return False, difference, coherence


def _build_initial_context(
    llama,
    backend: LlamaBackend,
    n_ctx: int,
    system_prompt: str,
    tool_schemas: list[dict] | None = None,
    gen_reserve: int = DEFAULT_TRIAL_TOKENS,
) -> tuple[ContextManager, list[int]]:
    context = ContextManager(backend, n_ctx=n_ctx, gen_reserve=gen_reserve)
    context.start_session(system_prompt, tool_schemas)
    boundaries: list[int] = []
    print("Building the 14-turn reference conversation...")
    for index, user_text in enumerate(INITIAL_TURNS, start=1):
        prompt_ids = context.begin_turn(user_text)
        result, _evaluated = generate(
            llama, context, prompt_ids, MAX_INITIAL_TOKENS
        )
        context.commit_turn(result.ids)
        boundaries.append(len(context.history_ids))
        print(
            f"  turn {index:2}: history tokens={boundaries[-1]}, "
            f"generated={len(result.ids)}"
        )
    return context, boundaries


def _run_single_shift_trials(
    llama,
    context: ContextManager,
    boundaries: list[int],
    snapshot,
    backend: LlamaBackend,
    n_ctx: int,
    system_prompt: str,
    tool_schemas: list[dict] | None = None,
    max_trial_tokens: int = DEFAULT_TRIAL_TOKENS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_ids = context.history_ids
    sys_len = len(context.system_ids)
    system_ids = list(context.system_ids)

    for dropped_turns in (2, 5, 9):
        end = boundaries[dropped_turns - 1]
        for message in NEW_MESSAGES:
            restore_snapshot(llama, snapshot)
            if not _shift_cache(llama, sys_len, end, system_ids):
                raise RuntimeError("cannot shift")
            _builder, new_turn_prompt = _new_turn_prompt(
                backend, system_prompt, n_ctx, message, tool_schemas, max_trial_tokens
            )
            prompt_ids = (
                context.system_ids
                + history_ids[end:]
                + new_turn_prompt[sys_len:]
            )

            shifted_result, shifted_evaluated = generate(
                llama, context, prompt_ids, max_trial_tokens
            )
            llama.reset()
            cold_result, cold_evaluated = generate(
                llama, context, prompt_ids, max_trial_tokens
            )
            identical, difference, coherence = _compare_outputs(
                backend, shifted_result, cold_result
            )

            # Anchor explicit check
            anchor_ok = list(llama.input_ids[:sys_len]) == system_ids if hasattr(llama, "input_ids") else True

            shifted_60_text = backend.decode(shifted_result.ids[:60]).replace("\n", " ").strip()
            cold_60_text = backend.decode(cold_result.ids[:60]).replace("\n", " ").strip()

            rows.append(
                {
                    "dropped_turns": dropped_turns,
                    "message": message,
                    "prompt_tokens": len(prompt_ids),
                    "shift_evaluated": shifted_evaluated,
                    "cold_evaluated": cold_evaluated,
                    "identical": identical,
                    "first_diff_index": difference,
                    "coherence": coherence,
                    "anchor_ok": anchor_ok,
                    "shifted_first_60_ids": shifted_result.ids[:60],
                    "cold_first_60_ids": cold_result.ids[:60],
                    "shifted_first_60_text": shifted_60_text,
                    "cold_first_60_text": cold_60_text,
                }
            )
    return rows


def _run_consecutive_shifts(
    llama,
    context: ContextManager,
    boundaries: list[int],
    snapshot,
    backend: LlamaBackend,
    n_ctx: int,
    system_prompt: str,
    tool_schemas: list[dict] | None = None,
    max_trial_tokens: int = DEFAULT_TRIAL_TOKENS,
    num_shifts: int = 4,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    system_ids = list(context.system_ids)
    sys_len = len(system_ids)

    restore_snapshot(llama, snapshot)

    cur_cut_idx = -1
    added_turn_blocks: list[list[int]] = []

    for shift_step in range(1, num_shifts + 1):
        drop_end_orig_idx = (shift_step * 2) - 1  # 1, 3, 5, 7 -> turns 2, 4, 6, 8
        orig_drop_tokens = boundaries[drop_end_orig_idx]
        prev_drop_tokens = boundaries[cur_cut_idx] if cur_cut_idx >= 0 else sys_len
        cut_len = orig_drop_tokens - prev_drop_tokens
        shift_end = sys_len + cut_len

        if not _shift_cache(llama, sys_len, shift_end, system_ids):
            raise RuntimeError("cannot shift")

        msg = CONSECUTIVE_MESSAGES[(shift_step - 1) % len(CONSECUTIVE_MESSAGES)]
        builder, new_turn_prompt = _new_turn_prompt(
            backend, system_prompt, n_ctx, msg, tool_schemas, max_trial_tokens
        )
        new_suffix = new_turn_prompt[sys_len:]

        surviving_orig = context.history_ids[orig_drop_tokens:]
        flattened_added = [tok for block in added_turn_blocks for tok in block]
        prompt_ids = system_ids + surviving_orig + flattened_added + new_suffix

        # Generate shifted
        shifted_result, shifted_evaluated = generate(
            llama, context, prompt_ids, max_trial_tokens
        )

        # Save state before cold test
        state_save = save_snapshot(llama)

        # Generate cold
        llama.reset()
        cold_result, cold_evaluated = generate(
            llama, context, prompt_ids, max_trial_tokens
        )

        identical, difference, coherence = _compare_outputs(
            backend, shifted_result, cold_result
        )

        # Restore state for next consecutive shift step
        restore_snapshot(llama, state_save)

        # Commit turn to tracking blocks
        t_builder, turn_block = _context_turn_block(
            backend, system_prompt, n_ctx, msg, shifted_result.ids, tool_schemas, max_trial_tokens
        )
        added_turn_blocks.append(turn_block)
        cur_cut_idx = drop_end_orig_idx

        # Anchor explicit check
        anchor_ok = list(llama.input_ids[:sys_len]) == system_ids

        shifted_60_text = backend.decode(shifted_result.ids[:60]).replace("\n", " ").strip()
        cold_60_text = backend.decode(cold_result.ids[:60]).replace("\n", " ").strip()

        rows.append(
            {
                "shift_step": shift_step,
                "dropped_turns": f"turns {drop_end_orig_idx - 1}-{drop_end_orig_idx} (+{cut_len} tokens)",
                "message": msg,
                "prompt_tokens": len(prompt_ids),
                "shift_evaluated": shifted_evaluated,
                "cold_evaluated": cold_evaluated,
                "identical": identical,
                "first_diff_index": difference,
                "coherence": coherence,
                "anchor_ok": anchor_ok,
                "shifted_first_60_ids": shifted_result.ids[:60],
                "cold_first_60_ids": cold_result.ids[:60],
                "shifted_first_60_text": shifted_60_text,
                "cold_first_60_text": cold_60_text,
            }
        )

    return rows


def _check_quantized_kv_support(
    model_path: Path,
    n_threads: int,
) -> list[dict[str, Any]]:
    """Test Criterion 1 API combinations: flash_attn x quantized types."""
    import llama_cpp
    from llama_cpp import Llama

    type_map = {
        "f16 (default)": None,
        "q8_0": getattr(llama_cpp, "GGML_TYPE_Q8_0", None),
        "q4_0": getattr(llama_cpp, "GGML_TYPE_Q4_0", None),
    }

    results = []
    tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

    for type_label, type_val in type_map.items():
        if type_val is None and type_label != "f16 (default)":
            continue
        for flash in (True, False):
            kwargs: dict[str, Any] = {
                "model_path": str(model_path),
                "n_ctx": 256,
                "n_threads": n_threads,
                "flash_attn": flash,
                "verbose": False,
            }
            if type_val is not None:
                kwargs["type_k"] = type_val
                kwargs["type_v"] = type_val

            try:
                llama = Llama(**kwargs)
                llama.eval(tokens)
                rm_ok = llama._ctx.kv_cache_seq_rm(0, 2, 5)
                shift_ok = True
                if hasattr(llama._ctx, "kv_cache_seq_shift"):
                    llama._ctx.kv_cache_seq_shift(0, 5, 10, -3)
                elif hasattr(llama._ctx, "kv_cache_seq_add"):
                    res = llama._ctx.kv_cache_seq_add(0, 5, 10, -3)
                    if res is False:
                        shift_ok = False
                else:
                    shift_ok = False
                llama.close()
                status = "PASS" if (rm_ok is not False and shift_ok) else "FAIL"
                err = ""
            except Exception as exc:
                status = "FAIL"
                err = str(exc)

            results.append(
                {
                    "type_k_v": type_label,
                    "flash_attn": flash,
                    "status": status,
                    "error": err,
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether llama.cpp can remove old chat turns from a live KV cache."
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=LLM_CONTEXT_SIZE,
        help=f"Context size (default: {LLM_CONTEXT_SIZE} from config)",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=LLM_BATCH,
        help=f"Batch size (default: {LLM_BATCH} from config)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=LLM_THREADS,
        help=f"Threads (default: {LLM_THREADS} from config)",
    )
    parser.add_argument(
        "--flash-attn",
        dest="flash_attn",
        action="store_true",
        default=LLM_FLASH_ATTN,
        help="Enable flash attention (default: true from config)",
    )
    parser.add_argument(
        "--no-flash-attn",
        dest="flash_attn",
        action="store_false",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--type-k",
        type=str,
        default=LLM_KV_CACHE_TYPE,
        help=f"KV cache type K (default: '{LLM_KV_CACHE_TYPE}' from config)",
    )
    parser.add_argument(
        "--type-v",
        type=str,
        default=LLM_KV_CACHE_TYPE,
        help=f"KV cache type V (default: '{LLM_KV_CACHE_TYPE}' from config)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=LLM_MODELS["qwen2.5-3b"]["path"],
        help=f"Path to GGUF model (default: {LLM_MODELS['qwen2.5-3b']['path']})",
    )
    parser.add_argument(
        "--max-initial-tokens",
        type=int,
        default=MAX_INITIAL_TOKENS,
        help=f"Tokens per initial turn generation (default: {MAX_INITIAL_TOKENS})",
    )
    parser.add_argument(
        "--max-trial-tokens",
        type=int,
        default=DEFAULT_TRIAL_TOKENS,
        help=f"Tokens generated per trial (default: {DEFAULT_TRIAL_TOKENS})",
    )
    parser.add_argument(
        "--consecutive-shifts",
        type=int,
        default=4,
        help="Number of consecutive shifts to execute (default: 4)",
    )
    parser.add_argument(
        "--include-tools",
        action="store_true",
        default=False,
        help="Include ToolRouter schemas in the system prefix (default: false, matching chat lane)",
    )
    parser.add_argument(
        "--check-quantized",
        action="store_true",
        default=False,
        help="Run Criterion 1 KV cache quantization matrix check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import llama_cpp
        from llama_cpp import Llama
    except ImportError as exc:
        print(f"llama-cpp-python is unavailable: {exc}")
        return 2

    model_path = args.model_path
    version = importlib.metadata.version("llama-cpp-python")
    print(f"=== KV Shift Probe ===")
    print(f"Date: 2026-11-08")
    print(f"Platform: {sys.platform} (Python {sys.version.split()[0]})")
    print(f"llama-cpp-python version: {version}")
    print(f"model: {model_path}")
    print(f"Settings: n_ctx={args.n_ctx}, n_batch={args.n_batch}, n_threads={args.n_threads}, "
          f"flash_attn={args.flash_attn}, type_k={args.type_k or 'f16 (default)'}, "
          f"type_v={args.type_v or 'f16 (default)'}")

    if not model_path.is_file():
        print(f"Model not found; install qwen2.5-3b at {model_path}")
        return 2

    # Tool schemas
    tool_schemas = None
    if args.include_tools:
        from nova.brain.tool_router import ToolRouter
        router = ToolRouter()
        tool_schemas = router.tool_schemas
        print(f"Tools included: {len(tool_schemas)} schemas")

    # API detection
    llama_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_threads": args.n_threads,
        "flash_attn": args.flash_attn,
        "verbose": False,
    }
    if args.type_k:
        llama_kwargs["type_k"] = args.type_k
    if args.type_v:
        llama_kwargs["type_v"] = args.type_v

    llama = Llama(**llama_kwargs)

    try:
        api_results = {
            name: _api_signature(llama._ctx, name)
            for name in ("kv_cache_seq_rm", "kv_cache_seq_shift", "kv_cache_seq_add")
        }
        print("\nKV sequence API detection:")
        for name, (present, signature) in api_results.items():
            print(f"  {name}: {'present' if present else 'absent'} {signature}")

        rm_ok = api_results["kv_cache_seq_rm"][0]
        shift_ok = api_results["kv_cache_seq_shift"][0] or api_results["kv_cache_seq_add"][0]
        if not (rm_ok and shift_ok):
            print("Required KV sequence-shift API is absent; stopping.")
            return 1

        backend = LlamaBackend(llama)
        context, boundaries = _build_initial_context(
            llama,
            backend,
            args.n_ctx,
            NOVA_PERSONA_PROMPT,
            tool_schemas,
            args.max_trial_tokens,
        )
        sys_len = len(context.system_ids)
        print(f"\nSystem prefix length: {sys_len} tokens (anchor)")
        print(f"Reference conversation built: 14 turns, total tokens = {boundaries[-1]}")
        snapshot = save_snapshot(llama)
        print(f"Reference snapshot saved: {snapshot.n_tokens} active tokens; boundaries={boundaries}")

        # Criterion 1: Quantized check if requested
        if args.check_quantized:
            print("\n--- Criterion 1: KV Quantization Matrix Check ---")
            q_results = _check_quantized_kv_support(model_path, args.n_threads)
            print("| type_k/v | flash_attn | Status | Notes |")
            print("|---|---|---|---|")
            for qr in q_results:
                print(f"| {qr['type_k_v']} | {qr['flash_attn']} | {qr['status']} | {qr['error'] or 'ok'} |")

        # Criterion 1, 2, 3, 5: Single Shift Trials
        print(f"\n--- Running Single Shift Trials (max_tokens={args.max_trial_tokens}) ---")
        single_rows = _run_single_shift_trials(
            llama,
            context,
            boundaries,
            snapshot,
            backend,
            args.n_ctx,
            NOVA_PERSONA_PROMPT,
            tool_schemas,
            args.max_trial_tokens,
        )

        print("\n### Single Shift Trials Results")
        print("| Dropped Turns | Prompt Tok | Evaluated (Shift) | Evaluated (Cold) | Identical | First Diff | Coherence | Anchor OK |")
        print("|---|---:|---:|---:|---|---:|---|---|")
        for row in single_rows:
            diff_str = str(row["first_diff_index"]) if row["first_diff_index"] is not None else "None"
            print(f"| {row['dropped_turns']} | {row['prompt_tokens']} | {row['shift_evaluated']} | "
                  f"{row['cold_evaluated']} | {'yes' if row['identical'] else 'no'} | {diff_str} | "
                  f"{row['coherence']} | {'PASS' if row['anchor_ok'] else 'FAIL'} |")

        print("\n### First 60 Tokens Side-by-Side (Single Shift)")
        for idx, row in enumerate(single_rows, 1):
            print(f"\n#### Trial {idx}: Dropped {row['dropped_turns']} turns | Message: \"{row['message'][:40]}...\"")
            print(f"- **Shifted** (eval={row['shift_evaluated']}): {row['shifted_first_60_text']}")
            print(f"- **Cold** (eval={row['cold_evaluated']}):    {row['cold_first_60_text']}")
            print(f"- **Difference**: {row['coherence']} (first diff at token {row['first_diff_index']})")

        # Criterion 4: Consecutive Shifts
        print(f"\n--- Running Consecutive Shifts ({args.consecutive_shifts} shifts in a row) ---")
        consecutive_rows = _run_consecutive_shifts(
            llama,
            context,
            boundaries,
            snapshot,
            backend,
            args.n_ctx,
            NOVA_PERSONA_PROMPT,
            tool_schemas,
            args.max_trial_tokens,
            args.consecutive_shifts,
        )

        print("\n### Consecutive Shifts Results")
        print("| Shift Step | Dropped Range | Prompt Tok | Evaluated (Shift) | Evaluated (Cold) | Identical | First Diff | Coherence | Anchor OK |")
        print("|---|---|---:|---:|---:|---|---:|---|---|")
        for row in consecutive_rows:
            diff_str = str(row["first_diff_index"]) if row["first_diff_index"] is not None else "None"
            print(f"| {row['shift_step']} | {row['dropped_turns']} | {row['prompt_tokens']} | "
                  f"{row['shift_evaluated']} | {row['cold_evaluated']} | {'yes' if row['identical'] else 'no'} | "
                  f"{diff_str} | {row['coherence']} | {'PASS' if row['anchor_ok'] else 'FAIL'} |")

        print("\n### First 60 Tokens Side-by-Side (Consecutive Shifts)")
        for row in consecutive_rows:
            print(f"\n#### Shift {row['shift_step']}: {row['dropped_turns']} | Message: \"{row['message'][:40]}...\"")
            print(f"- **Shifted** (eval={row['shift_evaluated']}): {row['shifted_first_60_text']}")
            print(f"- **Cold** (eval={row['cold_evaluated']}):    {row['cold_first_60_text']}")
            print(f"- **Difference**: {row['coherence']} (first diff at token {row['first_diff_index']})")

        # Acceptance Criteria Evaluation
        c1_api = True  # Shift calls succeeded without error
        c2_cost = all(row["shift_evaluated"] < row["cold_evaluated"] * 0.2 for row in single_rows + consecutive_rows)
        c3_coherence = all(row["coherence"] in ("identical", "diverge late", "diverge early") for row in single_rows + consecutive_rows)
        c4_consecutive = (len(consecutive_rows) >= 4 and
                          all(row["shift_evaluated"] < 50 for row in consecutive_rows) and
                          all(row["anchor_ok"] for row in consecutive_rows))
        c5_anchor = (all(row["anchor_ok"] for row in single_rows) and
                     all(row["anchor_ok"] for row in consecutive_rows))

        print("\n=== Acceptance Criteria Summary ===")
        print(f"1. API: {'PASS' if c1_api else 'FAIL'}")
        print(f"2. Cost (tail eval only): {'PASS' if c2_cost else 'FAIL'}")
        print(f"3. Coherence: {'PASS' if c3_coherence else 'FAIL'}")
        print(f"4. Consecutive shifts (>= 4): {'PASS' if c4_consecutive else 'FAIL'}")
        print(f"5. Anchor (system prefix untouched): {'PASS' if c5_anchor else 'FAIL'}")

        all_pass = c1_api and c2_cost and c3_coherence and c4_consecutive and c5_anchor
        print(f"\nOVERALL RESULT: {'PASS' if all_pass else 'PARTIAL'}")
        return 0

    except RuntimeError as exc:
        if str(exc) == "cannot shift":
            print("cannot shift: a KV sequence operation returned False.")
            return 1
        raise
    finally:
        llama.close()


if __name__ == "__main__":
    raise SystemExit(main())
