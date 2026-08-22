"""
nova/memory/core_store.py

Operational SQLite state that is NOT part of the persistent-memory
"brain" (see nova/memory/vault_store.py for that). This store now only
holds:
  - conversation_log:   a rolling raw chat log (used by memory_agent for
                         short extraction context, and for debugging)
  - project_profiles:   ingestion pipeline's per-project summaries
  - project_ingest_rules: ingestion pipeline's cached ignore rules

The old entity-graph / flat-fact tables that used to store long-term
user memory here have been removed - that responsibility now belongs
entirely to the Obsidian-vault store in vault_store.py.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from nova.config import CORE_STORE_DB_PATH

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS project_profiles (
    project_id      TEXT PRIMARY KEY,
    summary_json    TEXT NOT NULL,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS project_ingest_rules (
    project_id      TEXT PRIMARY KEY,
    ignore_paths    TEXT NOT NULL,
    ignore_ext      TEXT NOT NULL,
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
    """Thin, thread-safe wrapper around the SQLite operational-state DB."""

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

    # ------------------------------------------------------------------
    # Project profiles / ingestion rules
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Conversation log
    # ------------------------------------------------------------------
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