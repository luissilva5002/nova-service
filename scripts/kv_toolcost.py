"""Count system-block tokens for the current tool schemas using the GGUF template."""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nova.config import LLM_MODELS  # noqa: E402

MODEL_ID = "qwen2.5-3b"
MODEL_PATH = LLM_MODELS[MODEL_ID]["path"]


def main() -> int:
    if not MODEL_PATH.is_file():
        print(
            f"Required model is missing: {MODEL_PATH}. "
            f"Place {MODEL_PATH.name} at the path configured for {MODEL_ID} and rerun.",
            file=sys.stderr,
        )
        return 2

    try:
        from llama_cpp import Llama
        from nova.brain.context_manager import LlamaBackend
        from nova.brain.intent_router import filter_tools_for_intent
        from nova.brain.llm_engine import LLMEngine
        import nova.skills as skills_package
    except ImportError as exc:
        print(f"Required runtime dependency is unavailable: {exc}", file=sys.stderr)
        return 2

    llama = Llama(model_path=str(MODEL_PATH), vocab_only=True, verbose=False)
    backend = LlamaBackend(llama)
    skill_tools: list[tuple[str, list[dict]]] = []
    for _, module_name, is_package in pkgutil.iter_modules(skills_package.__path__):
        if not is_package or module_name.startswith("_"):
            continue
        tools_module = importlib.import_module(f"nova.skills.{module_name}.tools")
        declared_tools = getattr(tools_module, "TOOLS")
        skill_tools.append(
            (
                module_name,
                [
                    {"type": "function", "function": schema}
                    for schema in declared_tools
                ],
            )
        )
    all_tools = [schema for _, schemas in skill_tools for schema in schemas]
    system_prompt = LLMEngine._build_system_prompt()

    memory_search = {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search memory for relevant context.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    rows: list[tuple[str, list[dict]]] = [("no tools", [])]
    for skill_name, schemas in skill_tools:
        rows.append((f"skill: {skill_name}", schemas))

    for intent in (
        "action_music",
        "action_calendar",
        "memory_write",
        "memory_recall",
        "chat",
    ):
        rows.append((f"intent: {intent}", filter_tools_for_intent(all_tools, intent)))

    rows.extend(
        [
            ("ALL tools", all_tools),
            ("memory_search", [memory_search]),
        ]
    )

    baseline = None
    print("| scope | tool names | system-block tokens | delta vs no tools |")
    print("|---|---|---:|---:|")
    try:
        for scope, schemas in rows:
            rendered = backend.render(
                [{"role": "system", "content": system_prompt}],
                schemas or None,
                add_generation_prompt=False,
            )
            token_count = len(backend.encode(rendered))
            if baseline is None:
                baseline = token_count
            names = ", ".join(
                schema["function"]["name"] for schema in schemas
            ) or "(none)"
            print(f"| {scope} | {names} | {token_count} | {token_count - baseline:+d} |")
    finally:
        llama.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
