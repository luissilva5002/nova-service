"""
nova/memory/core_store.py

Tier 2 memory (Short-Term Core Memory / "the organic self").
A tiny SQLite key-value + structured-fact store. This is what gets
injected into the static system-prompt prefix (KV-cached by llama.cpp),
so it must stay small (5-10 key/value pairs, target ~100-200 tokens).

Also stores per-project "profiles" (Dense Fact Compression /
Scoped Context Switching, see architecture doc) and the ingestion
rule cache (project_ingest_rules) used by the ingestion pipeline.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from nova.config import CORE_STORE_DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_facts (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS project_profiles (
    project_id      TEXT PRIMARY KEY,
    summary_json    TEXT NOT NULL,   -- compressed dense fact block, e.g. {"proj":..,"stack":..,"rules":[..]}
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS project_ingest_rules (
    project_id      TEXT PRIMARY KEY,
    ignore_paths    TEXT NOT NULL,   -- JSON array
    ignore_ext      TEXT NOT NULL,   -- JSON array
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class CoreStore:
    """Thin, thread-safe wrapper around the SQLite core memory DB."""

    def __init__(self, db_path: Path = CORE_STORE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------
    # Tier 2: flat user facts (active_project, name, preferences, ...)
    # ---------------------------------------------------------------
    def set_fact(self, key: str, value: Any) -> None:
        with _lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO user_facts (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
                (key, json.dumps(value)),
            )

    def get_fact(self, key: str, default: Any = None) -> Any:
        with _lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM user_facts WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def all_facts(self) -> dict:
        with _lock, self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM user_facts").fetchall()
        return {k: json.loads(v) for k, v in rows}

    # ---------------------------------------------------------------
    # Project profiles (Dense Fact Compression - loaded only when active)
    # ---------------------------------------------------------------
    def set_project_profile(self, project_id: str, summary: dict) -> None:
        with _lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO project_profiles (project_id, summary_json, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(project_id) DO UPDATE SET summary_json=excluded.summary_json, updated_at=CURRENT_TIMESTAMP",
                (project_id, json.dumps(summary)),
            )

    def get_project_profile(self, project_id: str) -> Optional[dict]:
        with _lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM project_profiles WHERE project_id=?", (project_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    # ---------------------------------------------------------------
    # Ingestion rule cache (Tier 2 of the ingestion pipeline - see
    # nova/ingestion/llm_filter.py). Avoids re-asking the LLM to
    # classify the same project tree twice.
    # ---------------------------------------------------------------
    def set_ingest_rules(self, project_id: str, ignore_paths: list, ignore_ext: list) -> None:
        with _lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO project_ingest_rules (project_id, ignore_paths, ignore_ext, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(project_id) DO UPDATE SET ignore_paths=excluded.ignore_paths, "
                "ignore_ext=excluded.ignore_ext, updated_at=CURRENT_TIMESTAMP",
                (project_id, json.dumps(ignore_paths), json.dumps(ignore_ext)),
            )

    def get_ingest_rules(self, project_id: str) -> Optional[dict]:
        with _lock, self._connect() as conn:
            row = conn.execute(
                "SELECT ignore_paths, ignore_ext FROM project_ingest_rules WHERE project_id=?",
                (project_id,),
            ).fetchone()
        if not row:
            return None
        return {"ignore_paths": json.loads(row[0]), "ignore_ext": json.loads(row[1])}

    # ---------------------------------------------------------------
    # Lightweight conversation log (optional, useful for debugging /
    # for the memory_agent to scan for new facts to extract)
    # ---------------------------------------------------------------
    def log_message(self, role: str, content: str) -> None:
        with _lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_log (role, content) VALUES (?, ?)", (role, content)
            )

    def recent_messages(self, limit: int = 20) -> list:
        with _lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM conversation_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"role": r, "content": c, "created_at": t} for r, c, t in reversed(rows)]


# Module-level singleton - import `core_store` elsewhere.
core_store = CoreStore()
