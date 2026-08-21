import logging

from nova.memory.core_store import core_store
from nova.skills.base_skill import BaseSkill
from nova.skills.remember.tools import TOOLS

logger = logging.getLogger("nova.skills.remember")


class RememberSkill(BaseSkill):
    @property
    def tools(self) -> list[dict]:
        return TOOLS

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name != "remember_fact":
            return "Unknown remember command."

        key = str(arguments.get("key") or "remembered_fact").strip()
        value = arguments.get("value")
        if not key or value is None:
            return "I need a key and a value to store this memory."

        try:
            core_store.set_fact(key, value)
            category = (arguments.get("category") or "memory").strip()
            logger.info("Stored remembered fact under key=%s category=%s", key, category)
            return f"Saved {key} to my memory."
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("Failed to save fact to memory")
            return f"I couldn't save that memory: {exc}"


skill = RememberSkill()
