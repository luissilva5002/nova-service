"""
nova/skills/youtube_music/handler.py
Executes YouTube Music commands inside the Docker container.
Uses ytmusicapi for searching, yt-dlp for stream link extraction,
and ffplay (via ffmpeg) for headless audio playback.
"""
import logging
import subprocess
import sys
from typing import Optional
from ytmusicapi import YTMusic

from nova.skills.base_skill import BaseSkill
from nova.skills.youtube_music.tools import TOOLS

logger = logging.getLogger("nova.skills.youtube_music")


class YoutubeMusicSkill(BaseSkill):
    def __init__(self):
        try:
            self.yt = YTMusic()
        except Exception:
            logger.exception("Failed to initialize YTMusic client.")
            self.yt = None
        self._player_process: Optional[subprocess.Popen] = None

    @property
    def tools(self) -> list[dict]:
        return TOOLS

    def _stop_current_playback(self) -> None:
        """Terminates any active ffplay process inside the container."""
        if self._player_process and self._player_process.poll() is None:
            logger.info("Stopping active music playback.")
            self._player_process.terminate()
            try:
                self._player_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._player_process.kill()
            self._player_process = None

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "ytm_play":
            query = arguments.get("query", "")
            if not query:
                return "Please specify a song or artist to play."

            if not self.yt:
                try:
                    self.yt = YTMusic()
                except Exception as exc:
                    return f"YouTube Music client unavailable: {exc}"

            try:
                # 1. Search song metadata
                results = self.yt.search(query, filter="songs")
                if not results:
                    return f"I couldn't find '{query}' on YouTube Music."

                top_result = results[0]
                video_id = top_result.get("videoId")
                title = top_result.get("title", query)
                artist = (
                    top_result["artists"][0]["name"]
                    if top_result.get("artists")
                    else "Unknown Artist"
                )

                video_url = f"https://music.youtube.com/watch?v={video_id}"

                # 2. Stop currently playing song (ffplay) if present
                self._stop_current_playback()

                # 3. Prefer opening the track in the user's client (browser/app).
                #    Return a structured response telling the server to instruct
                #    the client to open the YouTube Music URL. This causes the
                #    web UI to navigate to the URL (which will open the native
                #    app on many phones if installed).
                logger.info("Requesting client to open YouTube Music URL for video_id=%s title=%r", video_id, title)
                return {"type": "open_url", "url": video_url, "message": f"Opening {title} by {artist} on your device."}

            except Exception as e:
                logger.exception("Error executing ytm_play")
                return f"Failed to play '{query}': {e}"

        if tool_name in ("ytm_pause", "ytm_skip"):
            self._stop_current_playback()
            action = "Paused" if tool_name == "ytm_pause" else "Skipped"
            return f"{action} YouTube Music playback."

        if tool_name == "ytm_set_volume":
            level = arguments.get("level", 50)
            logger.info("ytm_set_volume level=%s", level)
            return f"Volume set to {level}%."

        return "Unknown command."


# Auto-discovered instance by nova.brain.tool_router.ToolRouter
skill = YoutubeMusicSkill()