TOOLS = [
    {
        "name": "remember_fact",
        "description": "Persist a user fact, note, preference, or project detail into NOVA's durable memory so it can be recalled in later chats.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short memory key, like 'project_alpha' or 'favorite_song'."},
                "value": {"type": "string", "description": "The fact or note to store."},
                "category": {"type": "string", "description": "Optional category such as 'project', 'personal', or 'preference'."},
            },
            "required": ["key", "value"],
        },
    }
]
