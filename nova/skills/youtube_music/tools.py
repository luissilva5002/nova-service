"""
nova/skills/youtube_music/tools.py
Declarative tool definitions for YouTube Music control.
"""

TOOLS = [
    {
        "name": "ytm_play",
        "description": "Play a track, artist, or album on YouTube Music based on prompt intent.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song name, artist, or album query extracted from user prompt.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ytm_skip",
        "description": "Skip to the next track or stop current playback.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "ytm_pause",
        "description": "Pause active YouTube Music playback.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "ytm_set_volume",
        "description": "Set YouTube Music playback volume level.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "Volume level from 0 to 100"},
            },
            "required": ["level"],
        },
    },
]