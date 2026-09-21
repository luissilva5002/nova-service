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
    if category not in ("preferences", "knowledge", "user"):
        raise ValueError(f"Unknown vault category: {category!r}")
    if category == "preferences":
        root = PREFERENCES_DIR
    elif category == "knowledge":
        root = KNOWLEDGE_DIR
    else:
        root = PERSISTENT_MEMORY_VAULT_DIR / "user"
    slug = _slugify(box)
    if subcategory:
        return root / _slugify(subcategory) / f"{slug}.md"
    return root / f"{slug}.md"


def _paired_note_path(category: str, box: str) -> Path:
    """Path of the note in the OTHER category with the same box name, used
    for auto-linking (spec: paired preferences<->knowledge nodes)."""
    other = "preferences" if category == "knowledge" else "knowledge"
    return _note_path(other, box)


def _resolve_existing_note_path(path: Path) -> Path:
    """Resolve a note path even when the vault stores an extensionless file.

    Some notes in the Obsidian-style vault are persisted as plain filenames
    without a .md suffix, while the rest of the system expects .md paths.
    """
    if path.exists():
        return path
    alt = path.with_suffix("") if path.suffix == ".md" else path
    if alt.exists():
        return alt
    return path


def _read_raw(path: Path) -> Optional[str]:
    path = _resolve_existing_note_path(path)
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
    resolved = _resolve_existing_note_path(path)
    raw = _read_raw(resolved)
    if raw is None:
        title = box.replace("_", " ").strip().title()
        frontmatter = _default_frontmatter(title, category, tags)
        body = f"\n# {title}\n\n## Updates\n"
        return resolved, frontmatter, body
    frontmatter, body = _split_frontmatter(raw)
    if tags:
        existing = set(frontmatter.get("tags", []) or [])
        frontmatter["tags"] = sorted(existing | set(tags))
    return resolved, frontmatter, body


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
    path = _resolve_existing_note_path(_note_path(category, box, subcategory))
    return _read_raw(path)


def read_note_body(category: str, box: str, subcategory: Optional[str] = None) -> Optional[str]:
    raw = read_note(category, box, subcategory)
    if raw is None:
        return None
    _, body = _split_frontmatter(raw)
    return body.strip()


def list_notes(category: str) -> list[dict]:
    if category == "preferences":
        root = PREFERENCES_DIR
    elif category == "knowledge":
        root = KNOWLEDGE_DIR
    elif category == "user":
        root = PERSISTENT_MEMORY_VAULT_DIR / "user"
    else:
        return []
    if not root.exists():
        return []
    notes = []
    seen_paths: set[Path] = set()
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix not in {".md", ""}:
            continue
        real_path = _resolve_existing_note_path(path)
        if real_path in seen_paths:
            continue
        seen_paths.add(real_path)
        raw = _read_raw(real_path) or ""
        frontmatter, _ = _split_frontmatter(raw)
        note_name = real_path.stem or real_path.name
        notes.append({
            "box": note_name,
            "subcategory": real_path.parent.name if real_path.parent != root else None,
            "path": str(real_path),
            "title": frontmatter.get("title", note_name),
            "tags": frontmatter.get("tags", []),
        })
    return notes


_MEMORY_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "been", "being", "by", "can", "could",
    "create", "do", "does", "did", "doing", "for", "from", "good", "hello", "hey",
    "hi", "how", "i", "im", "in", "is", "it", "its", "just", "me", "memory",
    "morning", "my", "nova", "of", "on", "or", "our", "please", "remember",
    "save", "should", "so", "store", "sup", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "today", "up", "us", "was",
    "we", "what", "when", "where", "who", "why", "will", "with", "would",
    "write", "you", "your", "yours"
}


def _extract_memory_keywords(query: str) -> list[str]:
    """Remove small-talk and filler tokens so a greeting like 'how are you' does
    not trigger a memory lookup on unrelated knowledge."""
    query = (query or "").strip().lower()
    if not query:
        return []
    keywords = []
    for token in re.split(r"\W+", query):
        token = token.strip()
        if len(token) <= 2:
            continue
        if token in _MEMORY_STOPWORDS:
            continue
        keywords.append(token)
    return keywords


def search_notes(query: str, category: Optional[str] = None, max_results: int = 5) -> list[dict]:
    """Keyword search across vault notes, using WORD-BOUNDARY matching only.

    Previously this used plain substring checks (`kw in lower`), which
    meant a query containing "play" would match "display", "playback",
    "player", etc. anywhere in a note's body - causing completely
    unrelated notes (e.g. hardware docs) to be surfaced for commands
    like "play a song". Word-boundary regex matching fixes that; a
    keyword must appear as a whole word to count as a match.
    """
    query = (query or "").strip()
    if not query:
        return []
    if re.search(r"\b(?:remember|note that|save this|save to memory|create a memory|make a memory|store this in memory|write down)\b", query, re.I):
        return []
    keywords = _extract_memory_keywords(query)
    if not keywords:
        return []

    keyword_patterns = [re.compile(rf"\b{re.escape(kw)}\b", re.I) for kw in keywords]

    categories = [category] if category in ("preferences", "knowledge", "user") else ["preferences", "knowledge", "user"]
    ranked = []
    for cat in categories:
        for note in list_notes(cat):
            raw = _read_raw(Path(note["path"])) or ""
            title = note["title"]
            score = 0
            for pattern in keyword_patterns:
                if pattern.search(title):
                    score += 10
                score += len(pattern.findall(raw))
            if score <= 0:
                continue
            _, body = _split_frontmatter(raw)
            clean_body = body.strip().replace("\n", " ")
            snippet = clean_body[:280]
            ranked.append({
                "category": cat,
                "box": note["box"],
                "subcategory": note["subcategory"],
                "title": note["title"],
                "snippet": snippet,
                "full_text": clean_body[:4000],
                "_score": score,
            })
    ranked.sort(key=lambda item: item["_score"], reverse=True)
    for item in ranked:
        item.pop("_score", None)
    return ranked[:max_results]

def _markdown_id_for_path(path: Path) -> str:
    """Stable, human-readable reference for a markdown document.

    The id is derived from the vault-relative path so it remains deterministic
    across restarts and editor sessions while still being compact enough for
    client-side fetches and debugging.
    """
    try:
        rel = path.relative_to(PERSISTENT_MEMORY_VAULT_DIR).with_suffix("")
    except ValueError:
        rel = path.with_suffix("")
    return rel.as_posix().replace("/", "__").replace("\\", "__").strip("_") or "note"


def resolve_note_by_markdown_id(markdown_id: str) -> Optional[dict]:
    """Look up the note metadata for a markdown_id string."""
    if not markdown_id:
        return None
    for category in ("preferences", "knowledge", "user"):
        for note in list_notes(category):
            path = Path(note["path"])
            if _markdown_id_for_path(path) == markdown_id:
                return {**note, "category": category}
    return None


def get_markdown_document(markdown_id: str) -> Optional[dict]:
    """Fetch a note's body and metadata for the lightweight document API."""
    note = resolve_note_by_markdown_id(markdown_id)
    if note is None:
        return None
    raw = _read_raw(Path(note["path"])) or ""
    frontmatter, body = _split_frontmatter(raw)
    title = frontmatter.get("title") or note.get("title") or Path(note["path"]).stem
    updated = frontmatter.get("updated") or datetime.fromtimestamp(Path(note["path"]).stat().st_mtime, tz=timezone.utc).isoformat()
    return {
        "id": markdown_id,
        "title": title,
        "content": (body or "").strip() or "# " + title + "\n",
        "updated_at": updated,
    }


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
    full_context_blocks = []
    for match in knowledge_matches:
        full_text = (match.get("full_text") or match.get("snippet") or "").strip()
        if full_text:
            full_context_blocks.append(f"### {match['title']}\n{full_text}")

    return {
        "profile": get_profile_fields(),
        "preferences": combined_preferences,
        "relevant_knowledge": knowledge_matches,
        "knowledge_context": "\n\n".join(full_context_blocks),
    }


def get_memory_graph(max_nodes: int = 60) -> dict:
    """Return a folder-aware graph of the memory vault.

    The graph mixes two node kinds:
      - folder: a directory container
      - document: a markdown note

    Document nodes still include a stable markdown_id so clients can fetch the
    note content on demand, while folder nodes allow UI zooming into nested
    memory containers.
    """
    import math
    from hashlib import sha1

    nodes: list[dict] = []
    links: list[dict] = []
    folder_ids: dict[str, str] = {}
    title_to_id: dict[str, str] = {}

    def _id_for_rel(rel: str) -> str:
        return sha1(rel.encode("utf-8")).hexdigest()

    def _ensure_folder(category: str, rel_dir: str, depth: int = 0, parent_id: Optional[str] = None) -> str:
        key = f"{category}:{rel_dir or '.'}"
        if key in folder_ids:
            return folder_ids[key]
        folder_id = _id_for_rel(f"{category}:{rel_dir or '.'}")
        folder_ids[key] = folder_id
        label = Path(rel_dir).name if rel_dir and rel_dir != "." else category.title()
        nodes.append({
            "id": folder_id,
            "label": label,
            "kind": "folder",
            "type": category,
            "path": str(PERSISTENT_MEMORY_VAULT_DIR / category / rel_dir) if rel_dir else str(PERSISTENT_MEMORY_VAULT_DIR / category),
            "parent_id": parent_id,
            "depth": depth,
            "x": 500,
            "y": 360,
            "size": 140,
        })
        if parent_id:
            links.append({"source": parent_id, "target": folder_id, "label": "contains", "kind": "folder"})
        return folder_id

    for category in ("user", "preferences", "knowledge"):
        if category == "user":
            root_dir = PERSISTENT_MEMORY_VAULT_DIR / "user"
        elif category == "preferences":
            root_dir = PREFERENCES_DIR
        else:
            root_dir = KNOWLEDGE_DIR
        if not root_dir.exists():
            continue
        category_root_id = _ensure_folder(category, ".", depth=0)
        for path in sorted(root_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".md", ""}:
                continue
            rel = path.relative_to(root_dir)
            rel_str = rel.as_posix()
            parent_rel = rel.parent
            parent_id = category_root_id
            if parent_rel != Path("."):
                current_rel = Path(".")
                for part in parent_rel.parts:
                    current_rel = current_rel / part
                    parent_id = _ensure_folder(category, current_rel.as_posix(), depth=len(current_rel.parts), parent_id=parent_id)

            raw = _read_raw(path) or ""
            fm, body = _split_frontmatter(raw)
            title = (fm.get("title") if isinstance(fm, dict) else None) or path.stem or path.name
            doc_id = _id_for_rel(f"{category}:{rel_str}")
            title_to_id[title] = doc_id

            nodes.append({
                "id": doc_id,
                "label": title,
                "kind": "document",
                "type": category,
                "markdown_id": _markdown_id_for_path(path),
                "path": str(path),
                "parent_id": parent_id,
                "depth": max(1, len(rel.parts)),
                "excerpt": (body.strip().splitlines()[0] if body.strip() else "")[:220],
                "tags": fm.get("tags", []) if isinstance(fm, dict) else [],
                "facts": [],
                "x": 500,
                "y": 360,
                "size": 110,
            })
            links.append({"source": parent_id, "target": doc_id, "label": "contains", "kind": "document"})

    visible = [n for n in nodes if n.get("kind") in {"folder", "document"}]
    if not visible:
        return {"root": "user", "nodes": [], "links": []}

    folder_nodes = [n for n in visible if n["kind"] == "folder"]
    document_nodes = [n for n in visible if n["kind"] == "document"]
    count = len(visible)

    for index, node in enumerate(visible):
        angle = 2 * math.pi * index / max(count, 1)
        radius = 220 + (node.get("depth", 1) * 75)
        x = 500 + math.cos(angle) * radius
        y = 360 + math.sin(angle) * radius * 0.75
        if node["kind"] == "folder":
            x = 500 + (node.get("depth", 0) * 180) - 90
            y = 360 + (index % 5) * 70 - 140
        node["x"] = round(x, 2)
        node["y"] = round(y, 2)

    seen_links: set[tuple[str, str, str]] = set()
    for node in document_nodes:
        raw = _read_raw(Path(node["path"])) or ""
        for match in _WIKILINK_PATTERN.finditer(raw):
            target_title = match.group(1).strip()
            target_id = title_to_id.get(target_title)
            if target_id is None or target_id == node["id"]:
                continue
            key = (node["id"], target_id, "linked")
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append({"source": node["id"], "target": target_id, "label": "linked", "kind": "reference"})

    if len(nodes) > max_nodes:
        visible_ids = {n["id"] for n in nodes[:max_nodes]}
        nodes = nodes[:max_nodes]
        links = [link for link in links if link["source"] in visible_ids and link["target"] in visible_ids]

    return {"root": "user", "nodes": nodes, "links": links}