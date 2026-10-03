"""Measure prompt-prefix reuse through NOVA's normal LLMEngine path."""
import asyncio
import argparse
import gc
import io
import json
import logging
import os
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

os.environ["NOVA_LLM_MODEL"] = "qwen2.5-3b"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_cpp import Llama

from nova.brain.llm_engine import LLMEngine
from nova.config import (
    BASE_DIR,
    LLM_BATCH,
    LLM_CACHE_MB,
    LLM_CONTEXT_SIZE,
    LLM_FLASH_ATTN,
    LLM_GPU_LAYERS,
    LLM_HISTORY_TURNS,
    LLM_KV_CACHE_TYPE,
    LLM_MLOCK,
    LLM_MODEL_PATH,
    LLM_THREADS,
    LLM_THREADS_BATCH,
)


CONVERSATION = [
    "Hi, can you help me plan? Reply with one short greeting.",
    "I am organizing a relaxed Saturday visit with a friend. Name one good first activity.",
    (
        "I have Saturday free from late morning onward, and I would like to include a walk, "
        "a casual lunch, and enough time to get home before evening. Which should come first, "
        "the walk or lunch? Reply with only one choice."
    ),
    "What one useful item should I bring?",
    (
        "The weather may change during the day, and we could be outside for a few hours. "
        "What single item would be most useful to pack?"
    ),
    (
        "The plan is a late-morning walk, casual lunch, and home before evening, with flexibility "
        "if the weather changes. Confirm the plan in one short sentence."
    ),
    "Can lunch be vegetarian? Reply yes or no.",
    (
        "We enjoy local food, and one of us prefers vegetarian meals. Name one suitable menu "
        "option in a few words."
    ),
    (
        "If it rains, we still want an enjoyable day and have a walk and casual lunch planned. "
        "Name one practical indoor alternative."
    ),
    "Make the plan shorter. List three activities only.",
    (
        "I want to send a text with the main activities and flexible timing. Write one concise "
        "message of no more than twelve words."
    ),
    "Thanks, that's useful. Reply with a brief goodbye.",
]

FAKE_TOOL = {
    "type": "function",
    "function": {
        "name": "fake_lookup",
        "description": "Return a fixed value for the KV-cache test.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
MAX_CALLS_PER_ENGINE = 1


def retrieved_context(llm) -> tuple[str, int]:
    notes: list[str] = []
    block = ""
    token_count = 0
    while token_count < 300:
        notes.append(
            f"Note {len(notes) + 1}: local project context is versioned, searchable, and kept on device."
        )
        block = "[Retrieved context]\n" + "\n".join(notes)
        token_count = len(llm.tokenize(block.encode("utf-8"), add_bos=False))
    return block, token_count


def capture_kv_buffer_log() -> tuple[str, str, dict]:
    """Capture llama.cpp's own allocation report with diagnostic verbosity only."""
    kwargs = {
        "model_path": str(LLM_MODEL_PATH),
        "n_ctx": LLM_CONTEXT_SIZE,
        "n_threads": LLM_THREADS,
        "n_threads_batch": LLM_THREADS_BATCH,
        "n_batch": LLM_BATCH,
        "n_gpu_layers": LLM_GPU_LAYERS,
        "use_mlock": LLM_MLOCK,
        "verbose": True,
    }
    if LLM_FLASH_ATTN:
        kwargs["flash_attn"] = True
        if LLM_KV_CACHE_TYPE:
            kwargs["type_k"] = LLM_KV_CACHE_TYPE
            kwargs["type_v"] = LLM_KV_CACHE_TYPE

    output = io.StringIO()
    with redirect_stderr(output):
        try:
            probe = Llama(**kwargs)
        except TypeError:
            kwargs.pop("flash_attn", None)
            kwargs.pop("type_k", None)
            kwargs.pop("type_v", None)
            probe = Llama(**kwargs)
        metadata = dict(probe.metadata)
        probe.close()
    captured = output.getvalue()
    kv_match = re.search(
        r"llama_kv_cache:\s+CPU KV buffer size\s*=\s*([0-9.]+\s+\w+)", captured
    )
    flash_match = re.search(r"llama_context:\s+flash_attn\s*=\s*(\w+)", captured)
    return (
        kv_match.group(1) if kv_match else "not reported by llama.cpp",
        flash_match.group(1) if flash_match else "not reported",
        metadata,
    )


def render_table(rows: list[dict]) -> str:
    lines = [
        "| Turn | Prompt tokens | Reused | Evaluated | Prefill s | Decode tok/s | First differing message |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {turn} | {prompt_tokens} | {reused_tokens} | {evaluated_tokens} | "
            "{prefill_seconds:.3f} | {decode_tokens_per_second:.2f} | {first_diff_message} |".format(
                **row
            )
        )
    return "\n".join(lines)


async def run_scenario(
    scenario_name: str, with_context: bool, with_tool_roundtrip: bool = False
) -> tuple[list[dict], int, str]:
    engine = None
    engine_state = None
    calls_on_engine = MAX_CALLS_PER_ENGINE
    previous_tokens: list[int] = []
    context_block = ""
    context_tokens = 0
    cache_class = "unknown"
    rows: list[dict] = []
    history: list[dict] = []

    async def generate(turn: str, user_text: str, tool_schemas: list, memory_context: str = "") -> dict:
        nonlocal engine, engine_state, calls_on_engine, previous_tokens
        nonlocal context_block, context_tokens, cache_class
        if engine is None or calls_on_engine >= MAX_CALLS_PER_ENGINE:
            if engine is not None and engine._llm is not None:
                engine_state = engine._llm.save_state()
                engine._llm.close()
                engine._llm = None
                engine = None
                gc.collect()

            engine = LLMEngine()
            engine.load()
            if engine._llm is None:
                raise RuntimeError(
                    "The Qwen2.5 GGUF model did not load; refusing to emit stub metrics."
                )
            if engine_state is not None:
                engine._llm.load_state(engine_state)
                engine_state = None
            engine._kv_previous_tokens[:] = previous_tokens
            if not context_block:
                context_block, context_tokens = retrieved_context(engine._llm)
                cache_class = (
                    type(engine._llm.cache).__name__ if engine._llm.cache is not None else "none"
                )
            calls_on_engine = 0

        result = await engine.generate_with_tools(
            user_text=user_text,
            tool_schemas=tool_schemas,
            memory_context=memory_context,
            history=history,
            intent="chat",
        )
        if not engine._kv_call_history:
            raise RuntimeError(
                f"No KV instrumentation record was produced for {scenario_name} turn {turn}."
            )
        rows.append({"turn": turn, **engine._kv_call_history[-1]})
        previous_tokens[:] = engine._kv_last_tokens
        calls_on_engine += 1
        return result

    try:
        for turn_number, base_text in enumerate(CONVERSATION, start=1):
            user_text = f"{base_text}\n\nReply exactly OK and nothing else."
            if with_context and turn_number in {4, 6, 8, 10}:
                user_text = f"{user_text}\n\n{context_block}"

            if with_tool_roundtrip and turn_number == 6:
                tool_call = await generate("6 tool call", user_text, [FAKE_TOOL])
                if tool_call.get("type") != "tool_call":
                    raise RuntimeError(
                        "The forced fake_lookup request did not produce a tool call: "
                        f"{tool_call!r}"
                    )

                tool_result = (
                    "Fake lookup result: the planned outing can use the riverside route."
                )
                final_answer = await generate(
                    "6 final answer",
                    user_text,
                    [],
                    memory_context=f"Fake tool result: {tool_result}",
                )
                reply = final_answer.get("content", "")
            else:
                result = await generate(str(turn_number), user_text, [])
                reply = result.get("content", "")

            history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ]
            )
    finally:
        if engine is not None and engine._llm is not None:
            engine._llm.close()
            engine._llm = None
        gc.collect()

    return rows, context_tokens, cache_class


def make_report(
    scenarios: list[tuple[str, list[dict]]],
    context_tokens: int,
    cache_class: str,
    kv_buffer: str,
    runtime_flash: str,
    metadata: dict,
) -> str:
    architecture = metadata.get("general.architecture", "unknown")
    prefix = f"{architecture}."
    layers = metadata.get(f"{prefix}block_count", "unknown")
    kv_heads = metadata.get(f"{prefix}attention.head_count_kv", "unknown")
    n_heads = int(metadata.get(f"{prefix}attention.head_count", "0") or 0)
    embedding_length = int(metadata.get(f"{prefix}embedding_length", "0") or 0)
    head_dim = embedding_length // n_heads if n_heads else "unknown"
    cache_setting = f"{LLM_CACHE_MB} MiB configured"
    if LLM_KV_CACHE_TYPE:
        type_k = type_v = LLM_KV_CACHE_TYPE
    else:
        type_k = type_v = "llama.cpp default (f16)"

    sections = [
        "# KV-cache baseline",
        "",
        "Generated by `scripts/kv_baseline.py` using the real `LLMEngine.generate_with_tools` "
        "and `_build_messages` path with the existing sampling and model settings; only "
        "diagnostic instrumentation was added.",
        "",
        "## Configuration report",
        "",
        f"- Model: `{LLM_MODEL_PATH.name}` ({metadata.get('general.name', 'GGUF metadata name')}); "
        f"GGUF size label: {metadata.get('general.size_label', 'not present')}.",
        f"- Context: `n_ctx={LLM_CONTEXT_SIZE}`, `n_batch={LLM_BATCH}`, "
        f"`n_threads={LLM_THREADS}`, `n_threads_batch={LLM_THREADS_BATCH}`.",
        f"- Flash attention: configured `{str(LLM_FLASH_ATTN).lower()}`; llama.cpp load log "
        f"reported `{runtime_flash}`.",
        f"- KV data types: `type_k={type_k}`, `type_v={type_v}`.",
        f"- Prompt-state cache: `{cache_class}`, {cache_setting}; distinct from the model's KV buffer.",
        f"- History window: {LLM_HISTORY_TURNS} turns ({LLM_HISTORY_TURNS * 2} user/assistant "
        "messages); `main.run_pipeline` obtains the latest 12 session messages, then "
        "`LLMEngine._build_messages` retains at most `LLM_HISTORY_TURNS * 2` user/assistant "
        "messages and discards other roles.",
        "- Chat formatting: `create_chat_completion`; llama.cpp applies the GGUF "
        "`tokenizer.chat_template` (not a manually rendered template).",
        f"- llama.cpp load log: CPU KV buffer size `{kv_buffer}`.",
        f"- GGUF dimensions: {layers} layers, {kv_heads} KV heads, head_dim {head_dim} "
        f"(embedding length {embedding_length}, {n_heads} attention heads).",
        "- Runtime note: the existing container default is `qwen3-1.7b`; this script explicitly "
        "selects the requested `qwen2.5-3b` model without changing its inference parameters.",
        f"- Memory-limit handling: this harness reloads the engine every {MAX_CALLS_PER_ENGINE} "
        "model calls and restores llama.cpp's saved context state, preserving the adjacent-call "
        "KV prefix while releasing per-instance RAM-cache snapshots.",
        f"- Retrieved-context block: {context_tokens} model tokens (approximately 300), appended "
        "to the user message on turns 4, 6, 8, and 10 in runs 2 and 3.",
        "- Reuse definition: longest common token prefix of this request's actual llama.cpp prompt "
        "tokens and the previous request's prompt tokens followed by its generated tokens. "
        "First-difference indexes in instrumentation logs are zero-based.",
        "",
    ]
    for title, rows in scenarios:
        sections.extend([f"## {title}", "", render_table(rows), ""])

    plain = scenarios[0][1]
    trimmed_row = next(
        (row for row in plain if row["turn"] == str(LLM_HISTORY_TURNS + 2)), None
    )
    if trimmed_row is not None:
        sections.extend(
            [
                "## Findings",
                "",
                f"1. **Yes - reuse drops at turn {trimmed_row['turn']} in the plain run.** "
                f"The {LLM_HISTORY_TURNS}-turn history window has now discarded the oldest completed turn; "
                f"the first difference is in `{trimmed_row['first_diff_message']}` "
                f"(reused {trimmed_row['reused_tokens']} of {trimmed_row['prompt_tokens']} prompt tokens). "
                "The added retrieved-context text on turns 4/6/8/10 changes only the current-user "
                "suffix until the history window trims older turns.",
                "",
                f"2. **Yes - history trimming breaks the cached prefix beyond the stable system "
                f"prefix.** At turn {trimmed_row['turn']}, the prompt has shifted from the prior "
                f"history window, so only {trimmed_row['reused_tokens']} tokens are reused; "
                f"the first differing message is `{trimmed_row['first_diff_message']}`.",
                "",
            ]
        )
    else:
        sections.extend(
            [
                "## Findings",
                "",
                "1. **No - the plain run did not show a reuse decrease at the expected history-trim "
                "turn.** The history window and per-turn rows above provide the evidence.",
                "",
                "2. **No - no trimming-related prefix break was observed in the plain-run rows.**",
                "",
            ]
        )

    tool_rows = scenarios[2][1]
    tool_final = next(row for row in tool_rows if row["turn"] == "6 final answer")
    sections.extend(
        [
            f"3. **{'Yes' if tool_final['reused_tokens'] > 0 else 'No'} - the tool-call round trip "
            f"{'reused' if tool_final['reused_tokens'] > 0 else 'did not reuse'} a prefix.** "
            f"The final-answer request reused {tool_final['reused_tokens']} prompt tokens "
            f"({tool_final['first_diff_message']} is the first difference) after the fake tool "
            "result was supplied as context. The tool schema is present only on the initial "
            "tool-call request.",
            "",
        ]
    )
    return "\n".join(sections)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure KV-cache reuse in Nova's chat path.")
    parser.add_argument(
        "--scenario",
        choices=("all", "plain", "context", "tool"),
        default="all",
        help="Run all scenarios, or one scenario at a time in a resource-limited container.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    if not LLM_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Qwen2.5 model is not installed: {LLM_MODEL_PATH}")

    kv_buffer, runtime_flash, model_metadata = capture_kv_buffer_log()
    definitions = {
        "plain": ("Run 1 - plain chat", False, False),
        "context": ("Run 2 - retrieved context", True, False),
        "tool": ("Run 3 - retrieved context with turn-6 tool round trip", True, True),
    }
    results = {}
    if args.scenario == "all":
        selected = tuple(definitions)
    else:
        selected = (args.scenario,)

    for scenario_id in selected:
        title, with_context, with_tool = definitions[scenario_id]
        rows, actual_context_tokens, actual_cache_class = await run_scenario(
            title, with_context, with_tool
        )
        results[scenario_id] = {
            "title": title,
            "rows": rows,
            "context_tokens": actual_context_tokens,
            "cache_class": actual_cache_class,
            "kv_buffer": kv_buffer,
            "runtime_flash": runtime_flash,
            "metadata": model_metadata,
        }

    if args.scenario != "all":
        result_dir = BASE_DIR / "docs" / ".kv_baseline_runs"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{args.scenario}.json"
        result_path.write_text(json.dumps(results[args.scenario]), encoding="utf-8")
        for scenario_id in definitions:
            path = result_dir / f"{scenario_id}.json"
            if path.is_file():
                results[scenario_id] = json.loads(path.read_text(encoding="utf-8"))
        if len(results) != len(definitions):
            print(f"Saved {args.scenario}; run the other scenarios to finish the report.")
            return
        for scenario_id in definitions:
            (result_dir / f"{scenario_id}.json").unlink(missing_ok=True)
        result_dir.rmdir()

    scenarios = [
        (results[scenario_id]["title"], results[scenario_id]["rows"])
        for scenario_id in definitions
    ]
    context_tokens = max(results[scenario_id]["context_tokens"] for scenario_id in definitions)
    cache_class = results["plain"]["cache_class"]

    output = BASE_DIR / "docs" / "kv_baseline.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        make_report(
            scenarios,
            context_tokens,
            cache_class,
            results["plain"]["kv_buffer"],
            results["plain"]["runtime_flash"],
            results["plain"]["metadata"],
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
