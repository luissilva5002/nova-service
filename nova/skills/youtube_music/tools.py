"""
nova/skills/youtube_music/tools.py
Declarative tool definitions for YouTube Music control. The LLM reads
these descriptions and converts free-form speech ("put on Linkin Park",
"I'm in the mood for In The End") into a structured call - no keyword
matching or trigger phrases required.
"""

TOOLS = [
    {
        "name": "ytm_play",
        "description": "Play a track, artist, or album on YouTube Music.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Song name, artist, or album"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ytm_skip",
        "description": "Skip to the next track on YouTube Music.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "ytm_pause",
        "description": "Pause playback on YouTube Music.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "ytm_set_volume",
        "description": "Set YouTube Music playback volume.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "Volume level from 0 to 100"},
            },
            "required": ["level"],
        },
    },
]
