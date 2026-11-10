"""Recover a tool call from free-form model text.

Small local models sometimes emit the call as text instead of a structured
``tool_calls`` field, and often get the JSON braces wrong:

    <tool_call>{{"name": "x", "arguments": {...}}}</tool_call>   doubled braces
    <tool_call>{{"name": "x", "arguments": {"a": 1}}</tool_call>  unbalanced (3 open, 2 close)
    <tool_call>{"name": "x", "arguments": {"a": 1},}</tool_call>  trailing comma

``extract_tool_call`` tries progressively more tolerant strategies and returns
``{"name": str, "arguments": dict}`` or ``None``. When ``allowed_names`` is
given, a call to any other tool is rejected.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

_TAG = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S)
_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_NAME = re.compile(r'"name"\s*:\s*"([^"\\]+)"')
_ARGS_KEY = re.compile(r'"(?:arguments|parameters)"\s*:\s*')


def _scan_object(text: str, start: int) -> Optional[str]:
    """Return the JSON object starting at ``text[start] == '{'``.

    String contents are skipped, so braces inside values do not count. If the
    text ends before the object closes, the missing closing braces are added.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    if depth > 0 and not in_string:
        return text[start:] + "}" * depth
    return None


def _loads_object(candidate: str) -> Optional[dict]:
    for text in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize(parsed: dict) -> Optional[dict]:
    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = parsed.get("arguments", parsed.get("parameters", {}))
    if isinstance(arguments, str):
        loaded = _loads_object(arguments.strip())
        arguments = loaded if loaded is not None else {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name.strip(), "arguments": arguments}


def _candidates(raw: str) -> Iterable[dict]:
    """Yield parsed dicts, most faithful strategy first."""
    direct = _loads_object(raw)
    if direct is not None:
        yield direct

    first = raw.find("{")
    if first == -1:
        return
    # Count leading braces ("{{" or "{ {") and try dropping 0..n-1 of them.
    leading = 0
    index = first
    while index < len(raw) and raw[index] in "{ \t\r\n":
        if raw[index] == "{":
            leading += 1
        index += 1
    brace_positions = [i for i, char in enumerate(raw[first:index], start=first) if char == "{"]
    for skip in range(max(leading, 1)):
        start = brace_positions[skip] if skip < len(brace_positions) else first
        obj = _scan_object(raw, start)
        if obj is None:
            continue
        parsed = _loads_object(obj)
        if parsed is not None:
            yield parsed

    # Last resort: pull the name and the arguments object out separately.
    name_match = _NAME.search(raw)
    if name_match:
        arguments: dict = {}
        args_match = _ARGS_KEY.search(raw, name_match.end() - 1) or _ARGS_KEY.search(raw)
        if args_match:
            brace = raw.find("{", args_match.end())
            if brace != -1:
                obj = _scan_object(raw, brace)
                loaded = _loads_object(obj) if obj else None
                if loaded is not None:
                    arguments = loaded
        yield {"name": name_match.group(1), "arguments": arguments}


def extract_tool_call(
    content: str, allowed_names: Optional[Iterable[str]] = None
) -> Optional[dict]:
    if not isinstance(content, str) or not content.strip():
        return None
    match = _TAG.search(content)
    raw = (match.group(1) if match else content).strip()
    raw = _FENCE.sub("", raw).strip()
    allowed = None if allowed_names is None else set(allowed_names)
    for parsed in _candidates(raw):
        call = _normalize(parsed)
        if call is None:
            continue
        if allowed is not None and call["name"] not in allowed:
            continue
        return call
    return None
