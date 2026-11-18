"""Probe KV-C-lite (experiment 6) against the real GGUF: strip each turn's [Context] block from the KV.

Run from the repository root with the qwen2.5-3b GGUF installed:
    python scripts/kv_strip_probe.py
    python scripts/kv_strip_probe.py --n-ctx 1024
    python scripts/kv_strip_probe.py --out docs/kv_strip_probe_results.md

It runs one 8-turn chat twice through the production LaneEngine, with and
without stripping, and reports token counts only (no timings):

  * evaluated tokens per turn (strip vs no strip),
  * whether the KV cache equals the stored history after each turn,
  * for every turn from the second, three replies at temperature 0 for the
    identical prompt: from the stripped KV, from a cold context, and from a
    CONTROL context (history evaluated in one plain chunk, no shift at all).

Identical replies are not expected, for two different reasons:

  * batching noise: results depend on how the prompt was split into batches,
    so even the control diverges from a cold run on some turns;
  * conditioning (strip only): the cached states of the user text and the reply
    that follow the removed block were computed while the block was still in the
    sequence. Removing the tokens fixes the positions, not those states, so the
    stripped cache is an approximation of a cold evaluation of the clean history.

Strip therefore diverges more than the control. That is a property of the
technique, not a failure, so the verdict only fails on structural problems (not
stripped, KV differs from the history, no saving) and flags degenerate output
(empty or looping replies). Whether the replies are good enough is for you to
judge from the printed texts.

The verdict line says whether it is safe to enable NOVA_KV_STRIP_CONTEXT=1.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nova.config import (  # noqa: E402
    LLM_BATCH,
    LLM_CONTEXT_SIZE,
    LLM_FLASH_ATTN,
    LLM_KV_CACHE_TYPE,
    LLM_MODELS,
    LLM_THREADS,
    LLM_THREADS_BATCH,
    NOVA_PERSONA_PROMPT,
)
from nova.brain.generation import GenerationParams, TokenGenerator, stop_ids_for  # noqa: E402
from nova.brain.kv_state import restore_snapshot, save_snapshot  # noqa: E402
from nova.brain.lane_engine import LaneEngine  # noqa: E402

CONVERSATION = [
    "Why does the Moon appear to change shape during the month?",
    "How does sunlight reach the side of Earth facing away from the Sun?",
    "Why do objects look smaller when they are farther away?",
    "What causes clouds to form when warm air rises on a humid day?",
    "Why does a metal spoon feel colder than a wooden spoon in the same room?",
    "Why can a rainbow appear after rain?",
    "How can birds find their way across long distances?",
    "Why does water boil at a lower temperature on a high mountain?",
]

# Same shape as the chat lane's per-turn block (LLMEngine._lane_context_block with a
# session-summary memory context), so the stripped span has a realistic size.
CONTEXT_BLOCK = (
    "[Context]\nCurrent date: Friday, 2026-11-13 (UTC+00:00)\n\nKnown user context: "
    + json.dumps(
        {
            "chat_context_summary": "The user is a computer engineering student asking short science questions.",
            "relevant_profile_fields": {},
        },
        ensure_ascii=False,
    )
    + "\n[/Context]"
)


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def classify(left: list[int], right: list[int]) -> tuple[str, int | None]:
    difference = first_difference(left, right)
    if difference is None:
        return "identical", None
    return ("diverge late" if difference >= 30 else "diverge early"), difference


EARLY = 30  # a divergence before this token index counts as early (replies are 48 tokens)


def is_early(difference: int | None) -> bool:
    return difference is not None and difference < EARLY


def is_degenerate(ids: list[int]) -> bool:
    """Empty, tiny or looping output: the signature of a broken cache."""
    return len(ids) < 4 or len(set(ids)) / len(ids) < 0.4


def run_conversation(llama, n_ctx: int, strip: bool, max_tokens: int, parity: bool) -> list[dict]:
    session = "kv-strip-probe"
    llama.reset()
    engine = LaneEngine(
        llama,
        n_ctx,
        NOVA_PERSONA_PROMPT,
        gen_reserve=max_tokens + 32,
        strip_context=strip,
    )
    params = GenerationParams(max_tokens=max_tokens, temperature=0.0)
    rows: list[dict] = []
    for turn, text in enumerate(CONVERSATION, start=1):
        cold_prompt = None
        history_before: list[int] = []
        if parity and turn >= 2:
            lane = engine._get_or_create_lane(session, None)
            history_before = list(lane.context.history_ids)
            cold_prompt = lane.context.begin_turn(text, CONTEXT_BLOCK, context_position="before")
            lane.context.abort_turn()

        result = engine.chat(session, text, CONTEXT_BLOCK, None, params)
        lane = engine._lanes[session]
        history = lane.context.history_ids
        n_tokens = int(llama.n_tokens)
        kv_equals_history = [int(t) for t in llama.input_ids[:n_tokens]] == history[:n_tokens]
        row = {
            "turn": turn,
            "prompt": result.metrics["prompt_tokens"],
            "reused": result.metrics["cache_hit_tokens"],
            "evaluated": result.metrics["evaluated_tokens"],
            "strip": result.metrics.get("kv_strip", "-"),
            "kv_equals_history": kv_equals_history,
            "ids": list(result.ids),
            "text": result.text,
            "parity": "-",
            "cold_evaluated": None,
            "first_diff": None,
            "control_diff": None,
            "control_parity": "-",
            "cold_text": "",
            "control_text": "",
        }
        if cold_prompt is not None:
            snapshot = save_snapshot(llama)
            generator = TokenGenerator(llama, stop_ids_for(lane.context, llama))
            llama.reset()
            cold = generator.generate(cold_prompt, params)
            # Control: the same clean history as one plain chunk (no shift), then the new turn.
            llama.reset()
            if cold_prompt[: len(history_before)] != history_before:
                raise RuntimeError("cold prompt does not start with the stored history")
            llama.eval(history_before)
            control = generator.generate(cold_prompt, params)
            restore_snapshot(llama, snapshot)
            row["parity"], row["first_diff"] = classify(list(result.ids), list(cold.ids))
            row["control_parity"], row["control_diff"] = classify(list(control.ids), list(cold.ids))
            row["cold_evaluated"] = cold.evaluated_tokens
            row["cold_text"], row["control_text"] = cold.text, control.text
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-ctx", type=int, default=LLM_CONTEXT_SIZE)
    parser.add_argument("--n-batch", type=int, default=LLM_BATCH)
    parser.add_argument("--n-threads", type=int, default=LLM_THREADS)
    parser.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=LLM_FLASH_ATTN)
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--no-parity", action="store_true", help="skip the cold-run comparison (faster)")
    parser.add_argument("--model-path", type=Path, default=LLM_MODELS["qwen2.5-3b"]["path"])
    parser.add_argument("--out", type=Path, default=None, help="also write the report to this Markdown file")
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        print(f"llama-cpp-python is unavailable: {exc}")
        return 2
    if not args.model_path.is_file():
        print(f"Model not found; install qwen2.5-3b at {args.model_path}")
        return 2

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# KV-C-lite probe (experiment 6)")
    emit()
    emit(f"- platform: {sys.platform} / {platform.machine()}, Python {sys.version.split()[0]}")
    emit(f"- llama-cpp-python: {importlib.metadata.version('llama-cpp-python')}")
    emit(f"- model: {args.model_path.name}")
    emit(
        f"- settings: n_ctx={args.n_ctx}, n_batch={args.n_batch}, n_threads={args.n_threads}, "
        f"flash_attn={args.flash_attn}, kv_type={LLM_KV_CACHE_TYPE or 'f16 (default)'}, max_tokens={args.max_tokens}"
    )
    emit()

    kwargs = dict(
        model_path=str(args.model_path),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_threads_batch=LLM_THREADS_BATCH,
        n_batch=args.n_batch,
        n_gpu_layers=0,
        verbose=False,
    )
    if args.flash_attn:
        kwargs["flash_attn"] = True
        if LLM_KV_CACHE_TYPE:
            kwargs["type_k"] = LLM_KV_CACHE_TYPE
            kwargs["type_v"] = LLM_KV_CACHE_TYPE
    llama = Llama(**kwargs)
    if hasattr(llama, "set_cache"):
        llama.set_cache(None)

    try:
        parity = not args.no_parity
        print("Running with strip ON ...", flush=True)
        on_rows = run_conversation(llama, args.n_ctx, True, args.max_tokens, parity)
        print("Running with strip OFF ...", flush=True)
        off_rows = run_conversation(llama, args.n_ctx, False, args.max_tokens, False)
    finally:
        llama.close()

    emit("| turn | prompt | evaluated (strip) | evaluated (no strip) | strip | KV == history | strip vs cold | first diff | control vs cold | first diff | cold evaluated |")
    emit("|---:|---:|---:|---:|---|---|---|---:|---|---:|---:|")
    for on, off in zip(on_rows, off_rows):
        def cell(value):
            return value if value is not None else "-"

        emit(
            f"| {on['turn']} | {on['prompt']} | {on['evaluated']} | {off['evaluated']} | {on['strip']} | "
            f"{'yes' if on['kv_equals_history'] else 'NO'} | {on['parity']} | {cell(on['first_diff'])} | "
            f"{on['control_parity']} | {cell(on['control_diff'])} | {cell(on['cold_evaluated'])} |"
        )
    emit()
    later = [row for row in on_rows if row["turn"] >= 2]
    off_later = [row for row in off_rows if row["turn"] >= 2]
    if later:
        mean_on = sum(row["evaluated"] for row in later) / len(later)
        mean_off = sum(row["evaluated"] for row in off_later) / len(off_later)
        emit(
            f"Mean evaluated tokens per turn from turn 2: {mean_on:.0f} with strip, {mean_off:.0f} without "
            f"({mean_off - mean_on:.0f} fewer)."
        )
        emit()
    if parity:
        emit("### Replies at temperature 0 (stripped KV, cold context, control without shift)")
        emit()
        for row in on_rows:
            if row["parity"] == "-":
                continue
            emit(f"#### Turn {row['turn']}")
            emit(f"- strip:   {row['text'].strip()[:220]}")
            emit(f"- cold:    {row['cold_text'].strip()[:220]}")
            emit(f"- control: {row['control_text'].strip()[:220]}")
            emit()

    structural_problems = []
    if not all(row["strip"] == "stripped" for row in on_rows):
        structural_problems.append("some turns were not stripped: " + ", ".join(sorted({r["strip"] for r in on_rows})))
    if not all(row["kv_equals_history"] for row in on_rows):
        structural_problems.append("KV differs from the stored history after a stripped turn")
    if not all(on["evaluated"] < off["evaluated"] for on, off in zip(on_rows[1:], off_rows[1:])):
        structural_problems.append("stripping did not reduce evaluated tokens on every turn from the second")
    compared = [row for row in on_rows if row["parity"] != "-"]
    strip_early = [row["turn"] for row in compared if is_early(row["first_diff"])]
    control_early = [row["turn"] for row in compared if is_early(row["control_diff"])]
    degenerate = [row["turn"] for row in on_rows if is_degenerate(row["ids"])]
    if compared:
        emit(
            f"Early divergence from the cold run (before token {EARLY}): strip on {len(strip_early)} of {len(compared)} turns, "
            f"control (no shift) on {len(control_early)} of {len(compared)}. The extra divergence of strip is expected "
            "(see the top of this script); judge it from the replies above."
        )
        emit()

    if structural_problems:
        verdict = "NO"
        reason = "; ".join(structural_problems)
    elif degenerate:
        verdict = "REVIEW"
        reason = f"empty, tiny or looping replies on turns {degenerate}; read the texts"
    else:
        verdict = "YES"
        reason = (
            "every turn was stripped, the KV matches the history token for token, evaluated tokens dropped and no reply "
            "is degenerate. The stripped cache approximates a cold evaluation, so skim the replies above"
        )
    emit(f"**Safe to enable NOVA_KV_STRIP_CONTEXT=1 on this machine: {verdict}** - {reason}.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0 if verdict != "NO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
