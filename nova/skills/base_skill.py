"""
nova/skills/base_skill.py

Standardized Plugin (Skill) Architecture. Every integration (YouTube
Music, system control, stocks, Flutter workspace introspection, ...)
is an isolated module under nova/skills/<name>/ with exactly two files:

    tools.py    Declarative tool definitions (OpenAI-compatible JSON
                schemas the LLM reads to know what it can call and
                with what parameters).
    handler.py  The actual Python logic that executes the local
                script or API request, plus a module-level `skill`
                instance the ToolRouter auto-discovers.

Adding a new integration should take under 10 minutes and require
zero changes to core brain logic (see architecture doc).
"""
from abc import ABC, abstractmethod


class BaseSkill(ABC):
    @property
    @abstractmethod
    def tools(self) -> list[dict]:
        """Returns a list of OpenAI-compatible function definitions, e.g.:
        [{"name": ..., "description": ..., "parameters": {...}}]"""
        raise NotImplementedError

    @abstractmethod
    def execute(self, tool_name: str, arguments: dict) -> str:
        """Executes `tool_name` with `arguments` and returns a short string
        confirmation for NOVA to speak back to the user."""
        raise NotImplementedError
