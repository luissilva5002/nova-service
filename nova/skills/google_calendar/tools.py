"""
nova/skills/google_calendar/tools.py
OpenAI-compatible tool schema definitions for Google Calendar operations.
"""

TOOLS = [
    {
        "name": "gc_list_today",
        "description": "List calendar events for today in the user's primary calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "calendar_id": {"type": "string", "description": "Optional calendar id (default: primary)"},
            },
            "required": []
        },
    },
    {
        "name": "gc_find",
        "description": "Find events matching a query (free text) in a given time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for in events (summary/description)."},
                "time_min": {"type": "string", "description": "RFC3339 timestamp for start of search range (optional)."},
                "time_max": {"type": "string", "description": "RFC3339 timestamp for end of search range (optional)."},
                "calendar_id": {"type": "string", "description": "Optional calendar id (default: primary)"},
            },
            "required": ["query"]
        },
    },
    {
        "name": "gc_create_event",
        "description": (
            "Create a calendar event. Provide summary, start/end (RFC3339) or "
            "start + duration_minutes. If the user states an explicit date "
            "(e.g. a specific day and month), use exactly that date - do NOT "
            "substitute today's date. Only use today's date if the user says "
            "'today' or gives no date at all. Compute relative dates (e.g. "
            "'tomorrow', 'Friday') using the upcoming-dates list provided in "
            "system context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "start": {"type": "string", "description": "RFC3339 start datetime (e.g. 2026-08-19T15:00:00+01:00)"},
                "end": {"type": "string", "description": "RFC3339 end datetime"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes (used if end not provided)"},
                "location": {"type": "string"},
                "calendar_id": {"type": "string", "description": "Optional calendar id (default: primary)"},
            },
            "required": ["summary", "start"]
        },
    },
    {
        "name": "gc_update_event",
        "description": "Update fields on an existing event. Provide event_id and fields to change.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "calendar_id": {"type": "string", "description": "Optional calendar id (default: primary)"},
            },
            "required": ["event_id"]
        },
    },
    {
        "name": "gc_delete_event",
        "description": "Delete an event by event_id.",
        "parameters": {"type": "object", "properties": {"event_id": {"type": "string"}, "calendar_id": {"type": "string"}}, "required": ["event_id"]},
    },
]
