"""
nova/ingestion/pipeline.py

Orchestrates the full 3-Step Dynamic Ingestion Workflow end-to-end:
  1. project_scanner.build_directory_tree()   - folder skeleton inspection
  2. llm_filter.classify_ignore_rules()        - LLM policy decision (cached)
  3. chunk + embed surviving files              - vector vectorization

Exposed to the API as POST /api/ingest/{project_id} in nova/main.py.
"""
import logging
from pathlib import Path

from nova.config import HOST_PROJECTS_DIR
from nova.ingestion.llm_filter import classify_ignore_rules, path_is_ignored
from nova.ingestion.project_scanner import build_directory_tree, flatten_tree_paths, is_binary_file
from nova.memory.vector_store import vector_store

logger = logging.getLogger("nova.ingestion.pipeline")

CHUNK_LINES = 60  # simple fixed-size line chunking; swap for AST-aware chunking later if desired


async def ingest_project(project_id: str, llm_engine, projects_dir: Path = HOST_PROJECTS_DIR) -> dict:
    project_root = Path(projects_dir) / project_id
    if not project_root.exists():
        return {"status": "error", "detail": f"Project path not found: {project_root}"}

    tree = build_directory_tree(project_root)
    tree_paths = flatten_tree_paths(tree)

    rules = await classify_ignore_rules(project_id, tree_paths, llm_engine)

    ids, texts, metadatas = [], [], []
    indexed_files = 0
    skipped_files = 0

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(project_root))

        if path_is_ignored(rel_path, {"ignore_extensions": rules["ignore_extensions"], "ignore_paths": rules["ignore_paths"]}):
            skipped_files += 1
            continue
        if is_binary_file(path):
            skipped_files += 1
            continue

        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            skipped_files += 1
            continue

        for start in range(0, len(lines), CHUNK_LINES):
            chunk_lines = lines[start:start + CHUNK_LINES]
            if not any(l.strip() for l in chunk_lines):
                continue
            chunk_text = "\n".join(chunk_lines)
            chunk_id = f"{project_id}:{rel_path}:{start}"
            ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append({
                "project_id": project_id,
                "file_path": rel_path,
                "line_start": start + 1,
                "line_end": start + len(chunk_lines),
            })
        indexed_files += 1

    vector_store.upsert_many(ids, texts, metadatas)

    logger.info("Ingested project '%s': %s files indexed, %s skipped, %s chunks.", project_id, indexed_files, skipped_files, len(ids))
    return {
        "status": "ok",
        "project_id": project_id,
        "files_indexed": indexed_files,
        "files_skipped": skipped_files,
        "chunks_embedded": len(ids),
        "rules": rules,
    }
