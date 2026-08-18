"""
nova/skills/youtube_music/handler.py
Executes the YouTube Music tool calls. Replace the TODO sections with a
real client (e.g. `ytmusicapi`, or a local companion app on the same
network) - the schema/dispatch plumbing around it never needs to change.
"""
import logging

from nova.skills.base_skill import BaseSkill
from nova.skills.youtube_music.tools import TOOLS

logger = logging.getLogger("nova.skills.youtube_music")


class YoutubeMusicSkill(BaseSkill):
    @property
    def tools(self) -> list[dict]:
        return TOOLS

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "ytm_play":
            query = arguments.get("query", "")
            # TODO: call your YouTube Music API client here, e.g.:
            #   from ytmusicapi import YTMusic
            #   yt = YTMusic("oauth.json")
            #   results = yt.search(query, filter="songs")
            #   yt.play(results[0]["videoId"])  # or send to your device bridge
            logger.info("ytm_play query=%r", query)
            return f"Playing {query} on YouTube Music."

        if tool_name == "ytm_skip":
            logger.info("ytm_skip")
            return "Skipped to the next track."

        if tool_name == "ytm_pause":
            logger.info("ytm_pause")
            return "Paused YouTube Music."

        if tool_name == "ytm_set_volume":
            level = arguments.get("level", 50)
            logger.info("ytm_set_volume level=%s", level)
            return f"Volume set to {level}%."

        return "Unknown command."


# Auto-discovered by nova.brain.tool_router.ToolRouter
skill = YoutubeMusicSkill()
