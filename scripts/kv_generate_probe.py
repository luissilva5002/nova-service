"""Probe token-id generation and llama.cpp prefix reuse with ContextManager."""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nova.config import (  # noqa: E402
    LLM_CONTEXT_SIZE,
    LLM_MODELS,
    LLM_THREADS,
    NOVA_PERSONA_PROMPT,
)

MODEL_ID = "qwen2.5-3b"
MODEL_PATH = LLM_MODELS[MODEL_ID]["path"]
USER_TURNS = [
    "I'm planning a quiet weekend walk and would like to have lunch afterward.",
    "Make the walk easy, and I would prefer somewhere with trees.",
    "For lunch, I like simple vegetarian food.",
    "Can you summarize the plan we have discussed?",
]
MAX_GENERATED_TOKENS = 40


def _generate_one(llama, backend, prompt_ids: list[int], stop_id: int) -> dict:
    evaluated_before_first_token = 0
    first_token_seen = False
    generated_ids: list[int] = []
    tokens_before = int(llama.n_tokens)
    original_eval = llama.eval

    def counted_eval(token_ids, *args, **kwargs):
        nonlocal evaluated_before_first_token
        if not first_token_seen:
            evaluated_before_first_token += len(token_ids)
        return original_eval(token_ids, *args, **kwargs)

    llama.eval = counted_eval
    generator = None
    try:
        generator = llama.generate(prompt_ids, temp=0.0, reset=False)
        for token in generator:
            token_id = int(token)
            generated_ids.append(token_id)
            first_token_seen = True
            if token_id == stop_id or len(generated_ids) >= MAX_GENERATED_TOKENS:
                break
    finally:
        if generator is not None:
            generator.close()
        del llama.eval

    return {
        "prompt_tokens": len(prompt_ids),
        "evaluated_before_first": evaluated_before_first_token,
        "reuse_estimate": max(0, len(prompt_ids) - evaluated_before_first_token),
        "n_tokens_before": tokens_before,
        "generated_ids": generated_ids,
        "text": backend.decode(generated_ids),
    }


def _print_table(run_name: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{run_name}")
    print(
        "| turn | prompt tokens | evaluated before first token | "
        "reuse estimate | llm.n_tokens before | decoded generated text |"
    )
    print("|---|---:|---:|---:|---:|---|")
    for turn_name, row in rows:
        text = row["text"].replace("|", "\\|").replace("\n", "\\n")
        print(
            f"| {turn_name} | {row['prompt_tokens']} | "
            f"{row['evaluated_before_first']} | {row['reuse_estimate']} | "
            f"{row['n_tokens_before']} | {text} |"
        )
    variation = rows[-1][1]
    if variation["reuse_estimate"] == 0:
        print("First-100-token divergence check: measured zero reused prompt tokens.")
    else:
        print(
            "First-100-token divergence check: measured "
            f"{variation['reuse_estimate']} reused prompt tokens (expected zero)."
        )


def _run_probe(disable_ram_cache: bool) -> list[tuple[str, dict]]:
    from nova.brain.context_manager import ContextManager, LlamaBackend
    from nova.brain.llm_engine import LLMEngine

    engine = LLMEngine()
    engine._active_model_id = MODEL_ID
    engine.load()
    llama = engine._llm
    if llama is None:
        raise RuntimeError(f"LLMEngine.load() did not load the required model: {MODEL_PATH}")
    if not hasattr(llama, "set_cache"):
        raise RuntimeError("This llama-cpp-python build does not expose set_cache().")
    if disable_ram_cache:
        llama.set_cache(None)
        if getattr(llama, "cache", None) is not None:
            raise RuntimeError("llama.set_cache(None) did not disable the completion cache.")
    elif getattr(llama, "cache", None) is None:
        raise RuntimeError("LLMEngine.load() did not attach its configured LlamaRAMCache.")

    backend = LlamaBackend(llama)
    im_end_ids = backend.encode("<|im_end|>")
    if len(im_end_ids) != 1:
        raise RuntimeError(f"Expected <|im_end|> to encode to one token, got {im_end_ids!r}.")
    stop_id = im_end_ids[0]

    context = ContextManager(
        backend,
        n_ctx=LLM_CONTEXT_SIZE,
        gen_reserve=MAX_GENERATED_TOKENS,
    )
    context.start_session(NOVA_PERSONA_PROMPT)

    results: list[tuple[str, dict]] = []
    last_prompt: list[int] | None = None
    for turn_number, user_text in enumerate(USER_TURNS, start=1):
        prompt_ids = context.begin_turn(user_text)
        result = _generate_one(llama, backend, prompt_ids, stop_id)
        context.commit_turn(result["generated_ids"])
        results.append((str(turn_number), result))
        last_prompt = prompt_ids

    if last_prompt is None:
        raise RuntimeError("The fixed conversation did not generate any turns.")

    if len(last_prompt) < 100:
        raise RuntimeError("Turn-four prompt is too short for the 100-token divergence check.")
    vocab_size = int(llama.n_vocab())
    if vocab_size < 2:
        raise RuntimeError(f"Invalid model vocabulary size: {vocab_size}")
    changed_prompt = list(last_prompt)
    for index in range(100):
        replacement = (changed_prompt[index] + 1) % vocab_size
        if replacement == changed_prompt[index]:
            replacement = (replacement + 1) % vocab_size
        changed_prompt[index] = replacement

    changed_result = _generate_one(llama, backend, changed_prompt, stop_id)
    results.append(("turn 4 prefix changed", changed_result))

    try:
        llama.close()
    finally:
        engine._llm = None
    return results


def main() -> int:
    if not MODEL_PATH.is_file():
        print(
            f"STOP: required model is missing: {MODEL_PATH}. "
            f"Part 3 requires {MODEL_PATH.name}; no substitute will be loaded.",
            file=sys.stderr,
        )
        return 2

    try:
        import llama_cpp
        from nova.brain.llm_engine import LLMEngine
    except ImportError as exc:
        print(f"Required runtime dependency is unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        llama_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_version = getattr(llama_cpp, "__version__", "unknown")

    print(f"llama-cpp-python version: {llama_version}")
    print(f"Model: {MODEL_PATH}")
    print(f"n_threads: {LLM_THREADS}")
    print(f"Context size: {LLM_CONTEXT_SIZE}")

    cached_rows = _run_probe(disable_ram_cache=False)
    _print_table("Run A - LlamaRAMCache attached", cached_rows)

    uncached_rows = _run_probe(disable_ram_cache=True)
    _print_table("Run B - llm.set_cache(None)", uncached_rows)

    cached_turn_reuse = [row["reuse_estimate"] for _, row in cached_rows[:4]]
    uncached_turn_reuse = [row["reuse_estimate"] for _, row in uncached_rows[:4]]
    if any(cached_turn_reuse[1:]):
        reuse_summary = f"observed in turns {[i + 1 for i, count in enumerate(cached_turn_reuse) if i and count]}"
    else:
        reuse_summary = "not observed across turns 2-4"
    print(f"Conclusion (i): With token-id generate(), prefix reuse was {reuse_summary}.")
    if cached_turn_reuse == uncached_turn_reuse:
        print("Conclusion (ii): LlamaRAMCache did not change the measured per-turn reuse counts.")
    else:
        print(
            "Conclusion (ii): LlamaRAMCache changed measured reuse counts: "
            f"attached={cached_turn_reuse}; disabled={uncached_turn_reuse}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
