"""
nova/skills/flutter_workspace/tools.py
Codebase inspection schemas - lets NOVA search and read files inside a
registered Flutter (or any) project workspace on demand, rather than
stuffing whole projects into the context window (see architecture doc,
"Layer 2: Project Knowledge (On-Demand Tools)").
"""

TOOLS = [
    {
        "name": "search_codebase",
        "description": (
            "Semantic search over the active project's ingested source files. "
            "Use this to find where something is implemented before answering "
            "questions about the user's own code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for, e.g. 'login logic'"},
                "project_id": {"type": "string", "description": "Project identifier (defaults to active project)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a specific file path inside the active project workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path inside the project"},
            },
            "required": ["path"],
        },
    },
]
