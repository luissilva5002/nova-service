"""
nova/skills/flutter_workspace/handler.py
Implements search_codebase (Tier-3 vector RAG lookup) and read_file
(direct, sandboxed filesystem read) against the active project.
"""
import logging

from nova.config import HOST_PROJECTS_DIR
from nova.memory.core_store import core_store
from nova.memory.vector_store import vector_store
from nova.skills.base_skill import BaseSkill
from nova.skills.flutter_workspace.tools import TOOLS

logger = logging.getLogger("nova.skills.flutter_workspace")


class FlutterWorkspaceSkill(BaseSkill):
    @property
    def tools(self) -> list[dict]:
        return TOOLS

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "search_codebase":
            return self._search_codebase(arguments)
        if tool_name == "read_file":
            return self._read_file(arguments)
        return "Unknown command."

    def _active_project(self, arguments: dict) -> str | None:
        return arguments.get("project_id") or core_store.get_fact("active_project")

    def _search_codebase(self, arguments: dict) -> str:
        query = arguments.get("query", "")
        project_id = self._active_project(arguments)
        if not project_id:
            return "No active project is set - tell NOVA which project to work on first."

        hits = vector_store.query(query, project_id=project_id)
        if not hits:
            return f"No matches found for '{query}' in {project_id}."

        lines = []
        for hit in hits:
            meta = hit.get("metadata", {})
            lines.append(f"{meta.get('file_path', '?')} (lines {meta.get('line_start')}-{meta.get('line_end')})")
        return "Found matches in: " + "; ".join(lines)

    def _read_file(self, arguments: dict) -> str:
        rel_path = arguments.get("path", "")
        project_id = self._active_project(arguments)
        if not project_id:
            return "No active project is set."

        # Sandbox: resolve strictly inside the read-only host_projects mount.
        base = (HOST_PROJECTS_DIR / project_id).resolve()
        target = (base / rel_path).resolve()
        if not str(target).startswith(str(base)):
            logger.warning("Blocked path traversal attempt: %s", rel_path)
            return "That path is outside the project workspace - blocked."
        if not target.exists():
            return f"File not found: {rel_path}"

        try:
            content = target.read_text(errors="replace")
        except UnicodeDecodeError:
            return f"{rel_path} appears to be a binary file - cannot display as text."

        snippet = content[:2000]
        return f"Contents of {rel_path} (first 2000 chars):\n{snippet}"


# Auto-discovered by nova.brain.tool_router.ToolRouter
skill = FlutterWorkspaceSkill()
