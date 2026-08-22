"""
nova/memory/vault_store.py

PERSISTENT MEMORY storage engine: an Obsidian-compatible Markdown vault on
disk, replacing the old SQLite entity-graph/flat-fact persistence for
long-term memory. Two categories model a "digital twin" of durable
knowledge about the user:

  preferences/   non-tangible behavioral/procedural preferences
  knowledge/     tangible facts, user profile, subject-matter notes

This module is a pure file-I/O driver. OUT OF SCOPE: nova/memory/
session_memory.py (short-term, per-session JSON) is untouched and remains
the active conversational context store - this module never reads or
writes session state.
"""
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
import hashlib

from nova.config import PERSISTENT_MEMORY_VAULT_DIR

logger = logging.getLogger("nova.vault_store")

_lock = threading.RLock()

PREFERENCES_DIR = PERSISTENT_MEMORY_VAULT_DIR / "preferences"
KNOWLEDGE_DIR = PERSISTENT_MEMORY_VAULT_DIR / "knowledge"

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]#|]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n?", re.S)

_RESERVED_FRONTMATTER_KEYS = {"title", "category", "created", "updated", "tags", "linked_notes"}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "note"


def _note_path(category: str, box: str, subcategory: Optional[str] = None) -> Path:
    if category not in ("preferences", "knowledge"):
        raise ValueError(f"Unknown vault category: {category!r}")
    root = PREFERENCES_DIR if category == "preferences" else KNOWLEDGE_DIR
    slug = _slugify(box)
    if subcategory:
        return root / _slugify(subcategory) / f"{slug}.md"
    return root / f"{slug}.md"


def _paired_note_path(category: str, box: str) -> Path:
    """Path of the note in the OTHER category with the same box name, used
    for auto-linking (spec: paired preferences<->knowledge nodes)."""
    other = "preferences" if category == "knowledge" else "knowledge"
    return _note_path(other, box)


def _read_raw(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    match = _FRONTMATTER_PATTERN.match(raw)
    if not match:
        return {}, raw
    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        frontmatter = {}
    body = raw[match.end():]
    return frontmatter, body


def _render(frontmatter: dict, body: str) -> str:
    fm_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm_text}\n---\n{body}"


def _default_frontmatter(title: str, category: str, tags: Optional[list[str]] = None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "title": title,
        "category": category,
        "created": now,
        "updated": now,
        "tags": tags or [],
        "linked_notes": [],
    }


def _ensure_note(
    category: str, box: str, subcategory: Optional[str] = None, tags: Optional[list[str]] = None
) -> tuple[Path, dict, str]:
    path = _note_path(category, box, subcategory)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_raw(path)
    if raw is None:
        title = box.replace("_", " ").strip().title()
        frontmatter = _default_frontmatter(title, category, tags)
        body = f"\n# {title}\n\n## Updates\n"
        return path, frontmatter, body
    frontmatter, body = _split_frontmatter(raw)
    if tags:
        existing = set(frontmatter.get("tags", []) or [])
        frontmatter["tags"] = sorted(existing | set(tags))
    return path, frontmatter, body


def _append_entry(body: str, content: str) -> str:
    """Appends a dated entry rather than overwriting - preserves history
    on every write instead of destroying prior notes."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n- **{timestamp}**: {content.strip()}\n"
    if "## Updates" in body:
        return body + entry
    return body + "\n## Updates\n" + entry


def _link_notes(frontmatter: dict, linked_title: str) -> dict:
    links = set(frontmatter.get("linked_notes", []) or [])
    links.add(linked_title)
    frontmatter["linked_notes"] = sorted(links)
    return frontmatter


def _write_note(path: Path, frontmatter: dict, body: str) -> None:
    frontmatter["updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(_render(frontmatter, body), encoding="utf-8")


def _auto_link_pair(category: str, box: str, frontmatter: dict, body: str) -> str:
    """If a paired note exists in the other category with the same box
    name, insert bi-directional WikiLinks and update both notes'
    linked_notes frontmatter (spec: 'Inter-Connectivity')."""
    paired_path = _paired_note_path(category, box)
    title = frontmatter.get("title", box)
    if not paired_path.exists():
        return body

    paired_raw = _read_raw(paired_path) or ""
    paired_fm, paired_body = _split_frontmatter(paired_raw)
    paired_title = paired_fm.get("title", box)

    if f"[[{paired_title}]]" not in body:
        body = body.rstrip() + f"\n\nRelated: [[{paired_title}]]\n"
    _link_notes(frontmatter, paired_title)

    if f"[[{title}]]" not in paired_body:
        paired_body = paired_body.rstrip() + f"\n\nRelated: [[{title}]]\n"
    _link_notes(paired_fm, title)
    _write_note(paired_path, paired_fm, paired_body)

    return body


def write_preference(box: str, content: str, tags: Optional[list[str]] = None) -> Path:
    with _lock:
        path, frontmatter, body = _ensure_note("preferences", box, tags=tags)
        body = _append_entry(body, content)
        body = _auto_link_pair("preferences", box, frontmatter, body)
        _write_note(path, frontmatter, body)
        logger.info("vault_store: wrote preference box=%s", box)
        return path


def write_knowledge(
    box: str, content: str, subcategory: Optional[str] = None, tags: Optional[list[str]] = None
) -> Path:
    with _lock:
        path, frontmatter, body = _ensure_note("knowledge", box, subcategory=subcategory, tags=tags)
        body = _append_entry(body, content)
        body = _auto_link_pair("knowledge", box, frontmatter, body)
        _write_note(path, frontmatter, body)
        logger.info("vault_store: wrote knowledge box=%s subcategory=%s", box, subcategory)
        return path


def read_note(category: str, box: str, subcategory: Optional[str] = None) -> Optional[str]:
    return _read_raw(_note_path(category, box, subcategory))


def read_note_body(category: str, box: str, subcategory: Optional[str] = None) -> Optional[str]:
    raw = read_note(category, box, subcategory)
    if raw is None:
        return None
    _, body = _split_frontmatter(raw)
    return body.strip()


def list_notes(category: str) -> list[dict]:
    root = PREFERENCES_DIR if category == "preferences" else KNOWLEDGE_DIR
    if not root.exists():
        return []
    notes = []
    for path in sorted(root.rglob("*.md")):
        raw = _read_raw(path) or ""
        frontmatter, _ = _split_frontmatter(raw)
        notes.append({
            "box": path.stem,
            "subcategory": path.parent.name if path.parent != root else None,
            "path": str(path),
            "title": frontmatter.get("title", path.stem),
            "tags": frontmatter.get("tags", []),
        })
    return notes


def search_notes(query: str, category: Optional[str] = None, max_results: int = 5) -> list[dict]:
    """Simple case-insensitive keyword search across vault notes. No
    embeddings needed at this scale - a personal vault of markdown files."""
    query = (query or "").strip().lower()
    if not query:
        return []
    keywords = [w for w in re.split(r"\W+", query) if len(w) > 2]
    if not keywords:
        return []

    categories = [category] if category in ("preferences", "knowledge") else ["preferences", "knowledge"]
    results = []
    for cat in categories:
        for note in list_notes(cat):
            raw = _read_raw(Path(note["path"])) or ""
            lower = raw.lower()
            if any(kw in lower for kw in keywords) or any(kw in note["title"].lower() for kw in keywords):
                _, body = _split_frontmatter(raw)
                snippet = body.strip().replace("\n", " ")[:280]
                results.append({
                    "category": cat,
                    "box": note["box"],
                    "subcategory": note["subcategory"],
                    "title": note["title"],
                    "snippet": snippet,
                })
    return results[:max_results]


def get_profile_fields() -> dict:
    """Reads structured scalar facts (name, active_project, etc.) from
    knowledge/user_profile.md's YAML frontmatter - the vault-backed
    replacement for the old flat user_facts table."""
    raw = read_note("knowledge", "user_profile")
    if raw is None:
        return {}
    frontmatter, _ = _split_frontmatter(raw)
    return {k: v for k, v in frontmatter.items() if k not in _RESERVED_FRONTMATTER_KEYS}


def update_profile_fields(fields: dict) -> Path:
    """Merges scalar fields (e.g. {'name': 'Luis', 'active_project': 'NOVA'})
    into knowledge/user_profile.md's frontmatter for fast structured lookup."""
    with _lock:
        path, frontmatter, body = _ensure_note("knowledge", "user_profile")
        for key, value in fields.items():
            if key in _RESERVED_FRONTMATTER_KEYS:
                continue
            frontmatter[key] = value
        _write_note(path, frontmatter, body)
        logger.info("vault_store: updated profile fields %s", list(fields.keys()))
        return path


def get_context_snippet(query: str = "", max_preference_chars: int = 800, max_knowledge_notes: int = 3) -> dict:
    """Builds the compact context payload injected into the system prompt:
    all preference notes (meant to stay small per the architecture), plus
    targeted knowledge search results for the current query."""
    preference_blocks = []
    for note in list_notes("preferences"):
        raw = _read_raw(Path(note["path"])) or ""
        _, body = _split_frontmatter(raw)
        preference_blocks.append(f"### {note['title']}\n{body.strip()}")
    combined_preferences = "\n\n".join(preference_blocks)
    if len(combined_preferences) > max_preference_chars:
        combined_preferences = combined_preferences[:max_preference_chars].rstrip() + "..."

    knowledge_matches = search_notes(query, category="knowledge", max_results=max_knowledge_notes) if query else []

    return {
        "profile": get_profile_fields(),
        "preferences": combined_preferences,
        "relevant_knowledge": knowledge_matches,
    }


def get_memory_graph(max_nodes: int = 60) -> dict:
    """Builds a UI-friendly node/link graph from the vault's WikiLinks, for
    the /api/memory/graph endpoint (replaces the old SQLite entity graph).

    Notes on IDs: Use a stable deterministic id derived from the note path so
    clients can maintain node identity across restarts/edits.
    """
    import math
    from hashlib import sha1

    pref_notes = [{**n, "type": "preferences"} for n in list_notes("preferences")]
    know_notes = [{**n, "type": "knowledge"} for n in list_notes("knowledge")]
    all_notes = (pref_notes + know_notes)[:max_nodes]
    if not all_notes:
        return {"root": "user", "nodes": [], "links": []}

    # Deterministic id per-note: SHA1 of the note's vault-relative path
    def _note_id_for_path(path_str: str) -> str:
        try:
            p = Path(path_str)
            rel = str(p.relative_to(PERSISTENT_MEMORY_VAULT_DIR))
        except Exception:
            rel = path_str
        return sha1(rel.encode("utf-8")).hexdigest()

    title_to_id: dict[str, str] = {}
    path_map: dict[str, dict] = {}
    for note in all_notes:
        nid = _note_id_for_path(note["path"])
        title_to_id[note["title"]] = nid
        path_map[nid] = note

    nodes = []
    links = []
    seen_links: set[tuple[str, str]] = set()
    count = len(all_notes)

    for i, note in enumerate(all_notes):
        angle = 2 * math.pi * i / max(count, 1)
        radius = 260
        x = 500 + math.cos(angle) * radius
        y = 360 + math.sin(angle) * radius * 0.8
        note_id = title_to_id[note["title"]]

        raw = _read_raw(Path(note["path"])) or ""
        fm, body = _split_frontmatter(raw)
        excerpt = (body.strip().splitlines()[0] if body.strip() else "")[:240]
        tags = fm.get("tags", []) if isinstance(fm, dict) else []

        nodes.append({
            "id": note_id,
            "label": note["title"],
            "type": note["type"],
            "path": note.get("path"),
            "tags": tags,
            "facts": [],
            "excerpt": excerpt,
            "x": round(x, 2),
            "y": round(y, 2),
            "size": 100,
        })

        for match in _WIKILINK_PATTERN.finditer(raw):
            target_title = match.group(1).strip()
            target_id = title_to_id.get(target_title)
            if target_id is None or target_id == note_id:
                continue
            key = (note_id, target_id, "linked")
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append({"source": note_id, "target": target_id, "label": "linked", "metadata": {}})

    return {"root": "user", "nodes": nodes, "links": links}