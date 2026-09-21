"""
nova/skills/remember/tools.py
Tool schemas for NOVA's persistent Obsidian-vault memory: writing to
preferences/ and knowledge/, and reading/searching notes back.
"""

TOOLS = [
    {
        "name": "update_user_preference",
        "description": (
            "Save a non-tangible behavioral, relational, or procedural "
            "preference into NOVA's persistent memory vault - e.g. how the "
            "user likes to be addressed, preferred explanation depth, how "
            "they like to study a subject, communication style. Use a short "
            "topic 'box' name to group related preferences together (e.g. "
            "'communication', 'study_methods', 'flutter_learning_preferences')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "box": {"type": "string", "description": "Short topic name for this preference note, e.g. 'communication' or 'study_methods'."},
                "content": {"type": "string", "description": "The preference to save, in a clear sentence."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for this note."},
            },
            "required": ["box", "content"],
        },
    },
    {
        "name": "update_user_knowledge",
        "description": (
            "Save a tangible fact, user profile detail, course/topic note, "
            "or subject-matter reference into NOVA's persistent memory "
            "vault - e.g. identity details, a course the user is enrolled "
            "in, a Flutter code snippet, a reference fact. Use a short "
            "topic 'box' name to group related knowledge together (e.g. "
            "'user_profile', 'flutter_development', 'database_systems')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "box": {"type": "string", "description": "Short topic name for this knowledge note, e.g. 'flutter_development'."},
                "content": {"type": "string", "description": "The fact or note to save."},
                "subcategory": {"type": "string", "description": "Optional grouping folder, e.g. 'academics' for course notes."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for this note."},
            },
            "required": ["box", "content"],
        },
    },
    {
        "name": "recall_note",
        "description": "Search or read back a specific note from NOVA's persistent memory vault (preferences or knowledge) by topic box name.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["preferences", "knowledge"], "description": "Which vault category to look in."},
                "box": {"type": "string", "description": "Topic box name to read directly, if known."},
                "query": {"type": "string", "description": "Free-text search query, used if 'box' is not known/exact."},
            },
            "required": ["category"],
        },
    },
]