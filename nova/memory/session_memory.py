import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.config import DATA_DIR

SESSION_MEMORY_DIR = DATA_DIR / "session_memory"
SESSION_MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _session_file(session_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")
    return SESSION_MEMORY_DIR / f"{safe_id}.json"


def _empty_session(session_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "session_id": session_id,
        "history": [],
        "summary": "",
        "created_at": now,
        "updated_at": now,
    }


def load_session(session_id: str) -> dict[str, Any]:
    path = _session_file(session_id)
    if not path.exists():
        return _empty_session(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_session(session_id)
    data.setdefault("history", [])
    data.setdefault("session_id", session_id)
    data.setdefault("summary", "")
    data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    return data


def save_session(session: dict[str, Any]) -> None:
    session_id = session.get("session_id") or "default"
    path = _session_file(session_id)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def append_turn(session_id: str, role: str, content: str) -> None:
    session = load_session(session_id)
    session["history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    session["summary"] = build_session_summary(session)
    save_session(session)


def _compact_fact_like_lines(history: list[dict[str, Any]], limit: int = 8) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    def add_fact(value: str) -> None:
        text = str(value).strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        facts.append(text)

    for turn in history[-limit:]:
        if turn.get("role") != "user":
            continue
        text = str(turn.get("content", "")).strip()
        if not text:
            continue
        lower = text.lower()

        name_match = re.search(r"\b(?:my name is|i am|call me)\s+([a-z][a-z'\- ]{1,40})", text, re.I)
        if name_match:
            extracted = name_match.group(1).strip()
            if not extracted.lower().startswith("working "):
                add_fact(f"name: {extracted}")
                continue

        project_match = re.search(r"\b(?:working on|project|app|repo|codebase)\s+(?:is|:)?\s*([a-z0-9_./ -]{2,80})", text, re.I)
        if project_match:
            add_fact(f"project: {project_match.group(1).strip()}")
            continue

        if re.search(r"\bfavorite artist\b|\bartist\b.*\b(like|love|prefer)\b", lower):
            artist_match = re.search(r"(?:favorite artist|artist)\s*(?:is|:)?\s*([a-z0-9 .'-]{2,80})", text, re.I)
            if artist_match:
                add_fact(f"favorite artist: {artist_match.group(1).strip()}")
                continue

        if re.search(r"\bremember\b|\bsave\b|\bnote\b", lower):
            add_fact(f"reminder: {text[:120]}")
            continue

        if re.search(r"\bcalendar\b|\bmeeting\b|\bdinner\b|\bappointment\b|\bplan\b", lower):
            add_fact(f"topic: {text[:120]}")
            continue

        if text.endswith("?"):
            continue

        if len(text.split()) <= 10:
            add_fact(text)

    return facts


def build_session_summary(session: dict[str, Any]) -> str:
    history = session.get("history", [])
    if not history:
        return "No recent chat context yet."

    facts = _compact_fact_like_lines(history, limit=12)
    if facts:
        summary = "; ".join(facts)
        if len(summary) > 420:
            summary = summary[:417].rstrip() + "..."
        return summary

    user_turns = [str(turn.get("content", "")).strip() for turn in history if turn.get("role") == "user"]
    recent = user_turns[-4:]
    summary = " | ".join(filter(None, recent))
    if len(summary) > 260:
        summary = summary[:257].rstrip() + "..."
    return summary


def get_session_summary(session_id: str) -> str:
    return load_session(session_id).get("summary") or "No summary yet."


def get_context(session_id: str, limit: int = 20) -> list[dict[str, str]]:
    session = load_session(session_id)
    history = session.get("history", [])
    return history[-limit:]


def get_session_snapshot(session_id: str, limit: int = 20) -> dict[str, Any]:
    session = load_session(session_id)
    return {
        "session_id": session_id,
        "summary": session.get("summary") or "No summary yet.",
        "history": session.get("history", [])[-limit:],
        "updated_at": session.get("updated_at"),
    }


def clear_session(session_id: str) -> None:
    path = _session_file(session_id)
    if path.exists():
        path.unlink(missing_ok=True)


def clear_all_sessions() -> None:
    for path in SESSION_MEMORY_DIR.glob("*.json"):
        path.unlink(missing_ok=True)
