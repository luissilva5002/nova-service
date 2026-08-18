"""
nova/brain/tool_router.py

NOVA's "nervous system" (see architecture doc). Responsibilities:
  1. Schema Aggregation - on boot, scans nova/skills/ for every skill
     module, collects its OpenAI-compatible tool JSON schemas, and
     exposes them to the LLM brain.
  2. Intent Interception - when the LLM emits a tool_call instead of
     plain text, the router catches it.
  3. Skill Dispatching - matches the tool name to the owning skill and
     calls its handler.execute(tool_name, arguments).
  4. Response Feedback Loop - returns the handler's result so main.py
     can feed it back into the LLM for a natural-language confirmation.
"""
import importlib
import logging
import pkgutil
from typing import Optional

from nova.skills.base_skill import BaseSkill

logger = logging.getLogger("nova.tool_router")

SKILLS_PACKAGE = "nova.skills"


class ToolRouter:
    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}          # skill module name -> instance
        self._tool_to_skill: dict[str, BaseSkill] = {}    # tool name -> owning skill instance
        self._tool_schemas: list[dict] = []
        self._discovered = False

    def discover_skills(self) -> None:
        """Import every subpackage under nova/skills/ and register skills that
        subclass BaseSkill and expose a `skill` instance in their handler module."""
        if self._discovered:
            return
        import nova.skills as skills_pkg

        for _, module_name, is_pkg in pkgutil.iter_modules(skills_pkg.__path__):
            if not is_pkg or module_name.startswith("_"):
                continue
            try:
                handler_module = importlib.import_module(f"{SKILLS_PACKAGE}.{module_name}.handler")
            except ModuleNotFoundError:
                logger.warning("Skill '%s' has no handler.py - skipping.", module_name)
                continue

            skill_instance = getattr(handler_module, "skill", None)
            if skill_instance is None or not isinstance(skill_instance, BaseSkill):
                logger.warning(
                    "Skill '%s' handler.py must expose a `skill = <BaseSkill subclass>()` instance - skipping.",
                    module_name,
                )
                continue

            self._skills[module_name] = skill_instance
            for schema in skill_instance.tools:
                tool_name = schema["name"]
                self._tool_to_skill[tool_name] = skill_instance
                self._tool_schemas.append(self._to_openai_tool_schema(schema))

            logger.info("Registered skill '%s' with tools: %s", module_name, [t["name"] for t in skill_instance.tools])

        self._discovered = True

    @staticmethod
    def _to_openai_tool_schema(raw_schema: dict) -> dict:
        """Wraps a raw {"name","description","parameters"} dict in the
        OpenAI-style {"type": "function", "function": {...}} envelope
        expected by llama-cpp-python's chat-completion tool calling."""
        return {"type": "function", "function": raw_schema}

    @property
    def tool_schemas(self) -> list:
        self.discover_skills()
        return self._tool_schemas

    def dispatch(self, tool_name: str, arguments: dict) -> str:
        """Executes the tool and returns NOVA's confirmation string."""
        self.discover_skills()
        skill = self._tool_to_skill.get(tool_name)
        if skill is None:
            logger.error("No skill registered for tool '%s'", tool_name)
            return f"Unknown command: {tool_name}"
        try:
            return skill.execute(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001 - never let a skill crash the pipeline
            logger.exception("Skill execution failed for tool '%s'", tool_name)
            return f"Sorry, '{tool_name}' failed: {exc}"

    def list_skills(self) -> list:
        self.discover_skills()
        return [
            {"skill": name, "tools": [t["name"] for t in inst.tools]}
            for name, inst in self._skills.items()
        ]


tool_router = ToolRouter()
