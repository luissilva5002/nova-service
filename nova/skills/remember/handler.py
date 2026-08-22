"""
nova/skills/remember/handler.py
Writes to / reads from NOVA's persistent Obsidian-vault memory
(nova/memory/vault_store.py).
"""
import logging
import re

from nova.memory import vault_store
from nova.skills.base_skill import BaseSkill
from nova.skills.remember.tools import TOOLS

logger = logging.getLogger("nova.skills.remember")


def _compact_memory_fact(content: str) -> str:
    """Reduce a raw sentence to the shortest meaningful fact.

    This keeps the persistent vault readable and avoids saving giant quoted
    clauses exactly as the user spoke them.
    """
    text = str(content or "").strip()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()

    # simple named-entity patterns
    for pattern in [
        r"^my\s+name\s+is\s+(.+)$",
        r"^i\s+am\s+(.+)$",
        r"^i'm\s+(.+)$",
        r"^call\s+me\s+(.+)$",
        r"^i\s+study\s+at\s+(.+)$",
        r"^i\s+am\s+a\s+(.+)$",
    ]:
        match = re.match(pattern, text, re.I)
        if match:
            return match.group(1).rstrip(".?! ")

    if re.match(r"^i am a? (.+?) at (.+)$", text, re.I):
        match = re.match(r"^i am a? (.+?) at (.+)$", text, re.I)
        left = match.group(1).strip().rstrip(".?! ")
        right = match.group(2).strip().rstrip(".?! ")
        if left and right:
            return f"You are a {left} at {right}."

    if " at " in text and re.search(r"\b(student|studying|study|engineering)\b", text, re.I):
        return text.rstrip(".?! ")

    return text.rstrip(".?! ")


class RememberSkill(BaseSkill):
    @property
    def tools(self) -> list[dict]:
        return TOOLS

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "update_user_preference":
            return self._update_preference(arguments)
        if tool_name == "update_user_knowledge":
            return self._update_knowledge(arguments)
        if tool_name == "recall_note":
            return self._recall(arguments)
        return "Unknown remember command."

    def _update_preference(self, arguments: dict) -> str:
        box = str(arguments.get("box") or "").strip()
        content = str(arguments.get("content") or "").strip()
        if not box or not content:
            return "I need a topic and a preference to save."
        tags = arguments.get("tags") if isinstance(arguments.get("tags"), list) else None
        compact = _compact_memory_fact(content)
        try:
            vault_store.write_preference(box, compact, tags=tags)
            return "I'll be sure to remember that."
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("Failed to write preference note")
            return "I couldn't save that preference right now."

    def _update_knowledge(self, arguments: dict) -> str:
        box = str(arguments.get("box") or "").strip()
        content = str(arguments.get("content") or "").strip()
        if not box or not content:
            return "I need a topic and a fact to save."
        subcategory = arguments.get("subcategory") or None
        tags = arguments.get("tags") if isinstance(arguments.get("tags"), list) else None
        compact = _compact_memory_fact(content)
        try:
            vault_store.write_knowledge(box, compact, subcategory=subcategory, tags=tags)
            return "I'll be sure to remember that."
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("Failed to write knowledge note")
            return "I couldn't save that right now."

    def _recall(self, arguments: dict) -> str:
        category = str(arguments.get("category") or "").strip()
        box = arguments.get("box")
        query = arguments.get("query")

        if category not in ("preferences", "knowledge"):
            return "Please specify whether to look in preferences or knowledge."

        if box:
            body = vault_store.read_note_body(category, box)
            if body is None:
                return f"I don't have a '{box}' note in {category}."
            return body[:600]

        if query:
            results = vault_store.search_notes(query, category=category)
            if not results:
                return f"I couldn't find anything matching '{query}' in {category}."
            lines = [f"- {r['title']}: {r['snippet']}" for r in results]
            return "Found:\n" + "\n".join(lines)

        return "Please tell me a topic box name or a search query."


skill = RememberSkill()