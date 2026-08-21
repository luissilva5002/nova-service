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
        "description": (
            "Update fields on an existing event. You almost never know the "
            "real event_id from conversation alone - instead, describe the "
            "event using 'query' (its summary/title, e.g. 'DB exam') and "
            "optionally 'date' (YYYY-MM-DD, if the user mentioned one) so "
            "the correct event can be located automatically. Only pass "
            "event_id directly if you were explicitly told the exact ID "
            "string earlier in this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Exact Google Calendar event ID, only if already known."},
                "query": {"type": "string", "description": "Event title/summary to search for, if event_id is not known."},
                "date": {"type": "string", "description": "YYYY-MM-DD to narrow the search, if mentioned by the user."},
                "summary": {"type": "string", "description": "New title, if changing it."},
                "description": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "calendar_id": {"type": "string", "description": "Optional calendar id (default: primary)"},
            },
            "required": []
        },
    },
    {
        "name": "gc_delete_event",
        "description": (
            "Delete an event. You almost never know the real event_id from "
            "conversation alone - instead, describe the event using 'query' "
            "(its summary/title, e.g. 'DB exam') and optionally 'date' "
            "(YYYY-MM-DD, if the user mentioned one) so the correct event "
            "can be located automatically. Only pass event_id directly if "
            "you were explicitly told the exact ID string earlier in this "
            "conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Exact Google Calendar event ID, only if already known."},
                "query": {"type": "string", "description": "Event title/summary to search for, if event_id is not known."},
                "date": {"type": "string", "description": "YYYY-MM-DD to narrow the search, if mentioned by the user."},
                "calendar_id": {"type": "string"},
            },
            "required": []
        },
    },
]