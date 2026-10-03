"""Client for the knowledge category in WebObsidian's Agent API."""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import yaml

from nova.config import AGENT_API_BASE_URL, AGENT_API_KEY
from nova.memory.query_utils import MEMORY_WRITE_QUERY, extract_memory_keywords

logger = logging.getLogger("nova.knowledge_client")

_FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n?", re.S)
_KNOWLEDGE_PATH_PREFIXES = ("knowledge/", "ng-sports/")
_FULL_CONTENT_RESULTS = 3
# The live "00 Smasher.md" index is about 1 KB and should trigger linked-note retrieval.
_SHORT_INDEX_NOTE_CHAR_LIMIT = 1200
_MAX_LINKED_NOTES = 3
_KNOWLEDGE_CONTENT_CHAR_BUDGET = 2500


class KnowledgeApiError(RuntimeError):
    """Raised when WebObsidian cannot complete an Agent API request."""


def knowledge_path_for_box(box: str, subcategory: Optional[str] = None) -> str:
    def slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "note"

    parts = ["knowledge"]
    if subcategory:
        parts.append(slugify(subcategory))
    parts.append(f"{slugify(box)}.md")
    return PurePosixPath(*parts).as_posix()


def _validate_knowledge_path(path: str) -> str:
    if not isinstance(path, str) or "\\" in path:
        raise ValueError("Knowledge paths must be relative POSIX paths under knowledge/.")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or any(part in ("", ".", "..") for part in path.split("/"))
        or not any(path.startswith(prefix) for prefix in _KNOWLEDGE_PATH_PREFIXES)
    ):
        raise ValueError("Knowledge paths must be relative POSIX paths under a configured knowledge folder.")
    return parsed.as_posix()


def _request(
    method: str,
    endpoint: str,
    *,
    payload: Optional[dict] = None,
    not_found_ok: bool = False,
) -> Optional[dict]:
    if not AGENT_API_KEY:
        raise KnowledgeApiError(
            "AGENT_API_KEY is not configured. Create a WebObsidian API key with read, write, "
            "and search scopes and add it to Nova's .env."
        )
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"X-API-Key": AGENT_API_KEY, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{AGENT_API_BASE_URL}/{endpoint.lstrip('/')}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=15) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if exc.code == 404 and not_found_ok:
            return None
        raise KnowledgeApiError(
            f"WebObsidian Agent API returned HTTP {exc.code} for {method} {endpoint}"
            + (f": {detail[:500]}" if detail else "")
        ) from exc
    except URLError as exc:
        raise KnowledgeApiError(
            f"Could not reach WebObsidian Agent API at {AGENT_API_BASE_URL}: {exc.reason}"
        ) from exc

    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KnowledgeApiError(f"WebObsidian returned invalid JSON for {method} {endpoint}") from exc
    if not isinstance(result, dict):
        raise KnowledgeApiError(f"WebObsidian returned an invalid response shape for {method} {endpoint}.")
    return result


def _linked_note_path(source_path: str, link: str) -> Optional[str]:
    target = link.split("|", 1)[0].split("#", 1)[0].strip()
    if not target:
        return None
    target_path = PurePosixPath(target)
    if target_path.suffix.lower() not in {".md", ".markdown"}:
        target_path = target_path.with_suffix(".md")
    if not target_path.is_absolute() and len(target_path.parts) == 1:
        target_path = PurePosixPath(source_path).parent / target_path
    candidate = target_path.as_posix()
    try:
        return _validate_knowledge_path(candidate)
    except ValueError:
        return None


def search_knowledge(
    query: str,
    max_results: int = 5,
    full_content_results: int = _FULL_CONTENT_RESULTS,
) -> list[dict]:
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")
    if full_content_results < 1:
        raise ValueError("full_content_results must be at least 1.")
    query = (query or "").strip()
    if not query or MEMORY_WRITE_QUERY.search(query):
        return []

    search_text = re.sub(r"\b\w+:\S+", " ", query)
    keywords = extract_memory_keywords(search_text)
    if not keywords:
        return []
    params = urlencode({"q": query, "limit": 100})
    response = _request("GET", f"search?{params}")
    hits = response.get("hits") if isinstance(response, dict) else None
    if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
        raise KnowledgeApiError("WebObsidian search response did not contain a valid hits list.")
    results = [
        hit for hit in hits
        if isinstance(hit.get("path"), str)
        and any(hit["path"].startswith(prefix) for prefix in _KNOWLEDGE_PATH_PREFIXES)
    ][:max_results]

    budget_remaining = _KNOWLEDGE_CONTENT_CHAR_BUDGET
    fetched_paths = {hit["path"] for hit in results if isinstance(hit.get("path"), str)}
    top_note: Optional[dict] = None

    def attach_content(hit: dict) -> Optional[dict]:
        nonlocal budget_remaining
        path = hit.get("path")
        if not isinstance(path, str):
            return None
        note = read_knowledge(path)
        if note is None:
            return None
        content = note["content"][:budget_remaining]
        if content:
            hit["content"] = content
            budget_remaining -= len(content)
        return note

    if results:
        top_note = attach_content(results[0])

    top_path = top_note.get("path") if top_note else None
    links = top_note.get("links") if top_note else None
    if (
        isinstance(top_path, str)
        and isinstance(top_note.get("content"), str)
        and len(top_note["content"]) < _SHORT_INDEX_NOTE_CHAR_LIMIT
        and isinstance(links, list)
        and budget_remaining > 0
    ):
        linked_count = 0
        for link in links:
            if not isinstance(link, str):
                continue
            linked_path = _linked_note_path(top_path, link)
            if linked_path is None or linked_path in fetched_paths:
                continue
            fetched_paths.add(linked_path)
            linked_note = read_knowledge(linked_path)
            if linked_note is None:
                continue
            content = linked_note["content"][:budget_remaining]
            if not content:
                break
            results.append({
                "path": linked_note.get("path", linked_path),
                "title": linked_note.get("title") or linked_path,
                "score": 0,
                "tags": linked_note.get("tags", []),
                "snippet": content[:280],
                "content": content,
                "linked_from": top_path,
            })
            budget_remaining -= len(content)
            linked_count += 1
            if linked_count >= _MAX_LINKED_NOTES or budget_remaining == 0:
                break

    for hit in results[1:full_content_results]:
        if budget_remaining == 0:
            break
        attach_content(hit)

    return results


def read_knowledge(path: str) -> Optional[dict]:
    path = _validate_knowledge_path(path)
    result = _request("GET", f"notes/{quote(path, safe='/')}", not_found_ok=True)
    if result is None:
        return None
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise KnowledgeApiError(f"WebObsidian returned an invalid note response for {path}.")
    return result


def write_knowledge(path: str, content: str) -> None:
    path = _validate_knowledge_path(path)
    if not isinstance(content, str):
        raise TypeError("Knowledge note content must be a string.")
    _request("PUT", f"notes/{quote(path, safe='/')}", payload={"content": content})
    logger.info("Wrote knowledge note through WebObsidian: %s", path)


def append_knowledge_entry(path: str, content: str, tags: Optional[list[str]] = None) -> None:
    """Preserve the existing fact-accumulation behavior using full-note PUTs."""
    path = _validate_knowledge_path(path)
    existing = read_knowledge(path)
    raw = existing["content"] if existing is not None else ""
    match = _FRONTMATTER_PATTERN.match(raw)
    frontmatter_raw = match.group(0) if match else ""
    body = raw[match.end():] if match else raw

    now = datetime.now(timezone.utc).isoformat()
    if existing is None:
        frontmatter = {
            "title": PurePosixPath(path).stem.replace("_", " ").title(),
            "category": "knowledge",
            "created": now,
            "updated": now,
            "tags": tags or [],
            "linked_notes": [],
        }
        frontmatter_raw = f"---\n{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()}\n---\n"
    else:
        frontmatter = existing.get("frontmatter", {})
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        frontmatter = dict(frontmatter)
        if tags:
            frontmatter["tags"] = sorted(set(frontmatter.get("tags", []) or []) | set(tags))
        frontmatter["updated"] = now
        frontmatter_raw = f"---\n{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()}\n---\n"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n- **{timestamp}**: {content.strip()}\n"
    if "## Updates" not in body:
        body += "\n## Updates\n"
    write_knowledge(path, frontmatter_raw + body + entry)
