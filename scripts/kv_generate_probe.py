"""Probe token-id generation, prefix reuse and KV lane swapping with ContextManager.

COUNTS ONLY: no timings are recorded. Run from the repo root inside the runtime that
has llama-cpp-python and the real qwen2.5-3b GGUF:  python scripts/kv_generate_probe.py

Verified against llama-cpp-python 0.3.16 and 0.3.36 source:
  * Llama.generate() reuses the cached prefix ONLY when reset=True (the default).
    The last prompt token is always re-evaluated to refresh the logits.
  * generate() never touches Llama.cache (LlamaRAMCache is used by create_completion).
  * Llama.save_state() copies a scores array of (n_batch x n_vocab) floats, which is
    hundreds of MiB, on top of the KV state.
"""
from __future__ import annotations

import ctypes
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
MAX_GENERATED_TOKENS = 40
CHAT_TURNS = [
    "I'm planning a quiet weekend walk and would like to have lunch afterward.",
    "Make the walk easy, and I would prefer somewhere with trees.",
    "For lunch, I like simple vegetarian food.",
    "Can you summarize the plan we have discussed?",
]
ACTION_SYSTEM = "You control a music player. Answer with a tool call only."
ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ytm_play",
            "description": "Play a song or genre.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
ACTION_USER = "Play some jazz."


def mib(num_bytes: int) -> str:
    return f"{num_bytes / 1048576:.1f} MiB"


class EvalCounter:
    """Counts tokens passed to Llama.eval until stop_counting() is called."""

    def __init__(self, llama) -> None:
        self.llama = llama
        self.total = 0
        self.active = True
        self._orig = None

    def __enter__(self):
        self._orig = self.llama.eval

        def counted(tokens, *args, **kwargs):
            if self.active:
                self.total += len(tokens)
            return self._orig(tokens, *args, **kwargs)

        self.llama.eval = counted
        return self

    def __exit__(self, *exc):
        del self.llama.eval  # remove the instance override, restoring the class method

    def stop_counting(self) -> None:
        self.active = False


def generate_ids(llama, backend, prompt_ids: list[int], stop_id: int) -> dict:
    """Generate greedily from token ids. reset=True (default) enables prefix reuse."""
    n_tokens_before = int(llama.n_tokens)
    generated: list[int] = []
    with EvalCounter(llama) as counter:
        generator = llama.generate(prompt_ids, temp=0.0)
        try:
            for token in generator:
                if not generated:
                    counter.stop_counting()  # everything evaluated so far was prompt work
                token_id = int(token)
                generated.append(token_id)
                if token_id == stop_id or len(generated) >= MAX_GENERATED_TOKENS:
                    break
        finally:
            generator.close()
    return {
        "prompt_tokens": len(prompt_ids),
        "n_tokens_before": n_tokens_before,
        "evaluated": counter.total,
        "reused": len(prompt_ids) - counter.total,
        "generated_ids": generated,
        "text": backend.decode(generated),
    }


def _state_api():
    """Return (get_size, copy_out, copy_in) using the non-deprecated names when present."""
    import llama_cpp

    if hasattr(llama_cpp, "llama_state_get_data"):
        get_size = llama_cpp.llama_state_get_size

        def copy_out(ctx, buf, size):
            return int(llama_cpp.llama_state_get_data(ctx, buf, size))

        def copy_in(ctx, buf, size):
            return int(llama_cpp.llama_state_set_data(ctx, buf, size))
    else:  # older bindings
        get_size = llama_cpp.llama_get_state_size

        def copy_out(ctx, buf, size):
            return int(llama_cpp.llama_copy_state_data(ctx, buf))

        def copy_in(ctx, buf, size):
            return int(llama_cpp.llama_set_state_data(ctx, buf))

    return get_size, copy_out, copy_in


def light_save(llama) -> dict:
    """Save KV/context state WITHOUT llama-cpp-python's large scores array."""
    get_size, copy_out, _ = _state_api()
    ctx = llama._ctx.ctx
    size = int(get_size(ctx))
    buf = (ctypes.c_uint8 * size)()
    n_bytes = copy_out(ctx, buf, size)
    compact = (ctypes.c_uint8 * n_bytes)()
    ctypes.memmove(compact, buf, n_bytes)
    return {
        "state": bytes(compact),
        "size": n_bytes,
        "input_ids": llama.input_ids.copy(),
        "n_tokens": int(llama.n_tokens),
        "seed": llama._seed,
    }


def light_restore(llama, saved: dict) -> None:
    _, _, copy_in = _state_api()
    llama.input_ids = saved["input_ids"].copy()
    llama.n_tokens = saved["n_tokens"]
    llama._seed = saved["seed"]
    if hasattr(llama, "_requires_eval"):  # newer llama-cpp-python versions
        llama._requires_eval = True
    array = (ctypes.c_uint8 * saved["size"]).from_buffer_copy(saved["state"])
    if copy_in(llama._ctx.ctx, array, saved["size"]) != saved["size"]:
        raise RuntimeError("state restore returned an unexpected size")


def full_save(llama):
    return llama.save_state()


def full_restore(llama, state) -> None:
    llama.load_state(state)


def new_chat(context_cls, backend):
    context = context_cls(backend, n_ctx=LLM_CONTEXT_SIZE, gen_reserve=MAX_GENERATED_TOKENS)
    context.start_session(NOVA_PERSONA_PROMPT)
    return context


def chat_turn(llama, backend, context, user_text: str, stop_id: int) -> dict:
    prompt = context.begin_turn(user_text)
    row = generate_ids(llama, backend, prompt, stop_id)
    context.commit_turn(row["generated_ids"])
    return row


def action_turn(llama, backend, context_cls, stop_id: int) -> dict:
    """Stateless action lane: different system+tools prefix, nothing is kept."""
    context = context_cls(backend, n_ctx=LLM_CONTEXT_SIZE, gen_reserve=MAX_GENERATED_TOKENS)
    context.start_session(ACTION_SYSTEM, ACTION_TOOLS)
    prompt = context.begin_turn(ACTION_USER)
    row = generate_ids(llama, backend, prompt, stop_id)
    context.abort_turn()
    return row


def print_rows(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n### {title}")
    print("| step | prompt tokens | llm.n_tokens before | evaluated | reused | generated text |")
    print("|---|---:|---:|---:|---:|---|")
    for name, row in rows:
        text = row["text"].replace("|", "\\|").replace("\n", "\\n")
        print(
            f"| {name} | {row['prompt_tokens']} | {row['n_tokens_before']} | "
            f"{row['evaluated']} | {row['reused']} | {text} |"
        )


def main() -> int:
    if not MODEL_PATH.is_file():
        print(
            f"STOP: required model is missing: {MODEL_PATH}. No substitute will be loaded.",
            file=sys.stderr,
        )
        return 2
    try:
        import llama_cpp
        from nova.brain.context_manager import ContextManager, LlamaBackend
        from nova.brain.llm_engine import LLMEngine
    except ImportError as exc:
        print(f"Required runtime dependency is unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(llama_cpp, "__version__", "unknown")

    engine = LLMEngine()
    engine._active_model_id = MODEL_ID
    engine.load()
    llama = engine._llm
    if llama is None:
        raise RuntimeError(f"LLMEngine.load() did not load: {MODEL_PATH}")

    backend = LlamaBackend(llama)
    stop_ids = backend.encode("<|im_end|>")
    if len(stop_ids) != 1:
        raise RuntimeError(f"<|im_end|> should be one token, got {stop_ids!r}")
    stop_id = stop_ids[0]

    print(f"llama-cpp-python: {version}")
    print(f"model: {MODEL_PATH.name}  n_ctx: {LLM_CONTEXT_SIZE}  n_threads: {LLM_THREADS}")
    print(f"n_batch: {llama.n_batch}  Llama.cache attached: {type(llama.cache).__name__}")
    print("(generate() bypasses Llama.cache, so the RAM cache is irrelevant to these runs)")

    # ---- E1: uninterrupted chat, prefix reuse via generate(reset=True) -----------
    llama.reset()
    chat = new_chat(ContextManager, backend)
    control = []
    for index, text in enumerate(CHAT_TURNS, 1):
        control.append((f"chat {index}", chat_turn(llama, backend, chat, text, stop_id)))
    print_rows("E1: uninterrupted chat (prefix reuse with generate)", control)

    # ---- E2: prefix divergence ---------------------------------------------------
    last_prompt = chat.begin_turn("One more question: what should I bring?")
    chat.abort_turn()
    vocab = int(llama.n_vocab())
    changed = list(last_prompt)
    for i in range(min(100, len(changed))):
        changed[i] = (changed[i] + 1) % vocab
    divergence = generate_ids(llama, backend, changed, stop_id)
    print_rows("E2: first 100 prompt tokens changed", [("diverged", divergence)])

    # ---- E3/E4: lane swap ---------------------------------------------------------
    def swap_run(label: str, save_fn, restore_fn):
        llama.reset()
        chat_lane = new_chat(ContextManager, backend)
        rows = []
        for index in (1, 2):
            rows.append((f"chat {index}", chat_turn(llama, backend, chat_lane, CHAT_TURNS[index - 1], stop_id)))
        saved = save_fn(llama)
        rows.append(("action lane (different prefix)", action_turn(llama, backend, ContextManager, stop_id)))
        restore_fn(llama, saved)
        rows.append(("chat 3 after restore", chat_turn(llama, backend, chat_lane, CHAT_TURNS[2], stop_id)))
        rows.append(("chat 4", chat_turn(llama, backend, chat_lane, CHAT_TURNS[3], stop_id)))
        print_rows(label, rows)
        return saved, rows

    full_state, full_rows = swap_run("E3: lane swap with llama.save_state()/load_state()", full_save, full_restore)
    print(
        f"E3 state size: scores {mib(full_state.scores.nbytes)} + "
        f"llama_state {mib(full_state.llama_state_size)}"
    )
    light_state, light_rows = swap_run("E4: lane swap with light_save()/light_restore() (no scores array)", light_save, light_restore)
    print(f"E4 state size: llama_state {mib(light_state['size'])}")

    # ---- Conclusions --------------------------------------------------------------
    print("\n### Conclusions")
    follow_ups = [row for _, row in control[1:]]
    ok_reuse = all(row["reused"] >= row["n_tokens_before"] - 1 for row in follow_ups)
    print(f"(i) generate() prefix reuse across chat turns: {'YES' if ok_reuse else 'NO'} "
          f"(evaluated per follow-up turn: {[row['evaluated'] for row in follow_ups]})")
    print(f"(ii) divergence check: reused {divergence['reused']} of {divergence['prompt_tokens']} "
          f"(expected 0)")
    ctrl_text = control[2][1]["text"]
    for name, rows in (("E3 full state", full_rows), ("E4 light state", light_rows)):
        after = next(row for label, row in rows if label == "chat 3 after restore")
        same = after["text"] == ctrl_text
        print(f"(iii) {name}: chat 3 evaluated {after['evaluated']} tokens, "
              f"reused {after['reused']}, output identical to uninterrupted run: {same}")

    llama.close()
    engine._llm = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())