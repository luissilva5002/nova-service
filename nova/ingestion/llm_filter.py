"""
nova/ingestion/llm_filter.py

Step 2 of the 3-Step Dynamic Ingestion Workflow: "LLM Policy Decision".
Sends the lightweight directory tree (from project_scanner.py) to
NOVA's brain with a single classification prompt, and gets back a JSON
array of paths/extensions to exclude from vector indexing (build
artifacts, platform boilerplate, generated code, binaries).

The result is cached in core_store.project_ingest_rules (Rule Caching)
so this classification only has to run once per project.
"""
import json
import logging

from nova.config import DEFAULT_IGNORE_DIR_HINTS, DEFAULT_IGNORE_EXTENSIONS
from nova.memory.core_store import core_store

logger = logging.getLogger("nova.ingestion.llm_filter")

CLASSIFICATION_PROMPT = (
    "Analyze this directory tree for a software project. Return ONLY a JSON "
    "object with two keys: \"ignore_paths\" (glob-style path prefixes that "
    "are build artifacts, platform boilerplate, or generated code with zero "
    "context about the core application logic) and \"ignore_extensions\" "
    "(file extensions that are binary/generated assets, e.g. images, locks, "
    "compiled output). Do not include any explanation, only JSON."
)


async def classify_ignore_rules(project_id: str, tree_paths: list, llm_engine) -> dict:
    """
    Returns {"ignore_paths": [...], "ignore_extensions": [...]}.
    Falls back to DEFAULT_IGNORE_* + heuristic dir hints if the LLM is
    unavailable (STUB mode) or returns malformed JSON.
    """
    cached = core_store.get_ingest_rules(project_id)
    if cached:
        return {"ignore_paths": cached["ignore_paths"], "ignore_extensions": cached["ignore_ext"]}

    tree_text = "\n".join(tree_paths[:400])  # keep the classification prompt small
    fallback = {
        "ignore_paths": [f"{hint}/*" for hint in DEFAULT_IGNORE_DIR_HINTS],
        "ignore_extensions": DEFAULT_IGNORE_EXTENSIONS,
    }

    try:
        raw = await llm_engine.generate_raw(
            system_prompt=CLASSIFICATION_PROMPT,
            user_prompt=tree_text,
            max_tokens=512,
        )
        parsed = json.loads(raw.strip())
        ignore_paths = parsed.get("ignore_paths", []) or fallback["ignore_paths"]
        ignore_ext = parsed.get("ignore_extensions", []) or fallback["ignore_extensions"]
    except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        logger.warning("llm_filter: falling back to default heuristics (%s)", exc)
        ignore_paths, ignore_ext = fallback["ignore_paths"], fallback["ignore_extensions"]

    core_store.set_ingest_rules(project_id, ignore_paths, ignore_ext)
    return {"ignore_paths": ignore_paths, "ignore_extensions": ignore_ext}


def path_is_ignored(rel_path: str, rules: dict) -> bool:
    """Applies the cached/classified rules to a single relative path."""
    for ext in rules.get("ignore_extensions", []):
        if rel_path.endswith(ext):
            return True
    for prefix in rules.get("ignore_paths", []):
        prefix_clean = prefix.rstrip("/*")
        if rel_path.startswith(prefix_clean):
            return True
    return False
