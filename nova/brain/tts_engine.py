"""
nova/brain/tts_engine.py

The Mouth - Piper TTS. A dedicated, non-LLM neural (VITS) text-to-speech
engine. Synthesizes sentence-by-sentence and streams raw audio bytes back
to the client chunk-by-chunk over the WebSocket, so playback can begin
before the full response has finished generating (see architecture doc
"Streaming audio enabled").
"""
import logging
import re
import asyncio
from typing import AsyncIterator

from nova.config import PIPER_VOICE_MODEL_PATH, PIPER_VOICE_CONFIG_PATH
from nova.brain.sentence_splitter import split_sentences

logger = logging.getLogger("nova.tts_engine")

try:
    from piper import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIPER_AVAILABLE = False



class TTSEngine:
    def __init__(self):
        self._voice = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not _PIPER_AVAILABLE:
            logger.warning("piper-tts not installed - TTS engine running in STUB mode.")
            self._loaded = True
            return
        if not PIPER_VOICE_MODEL_PATH.exists():
            logger.warning(
                "No Piper voice model found at %s - TTS engine running in STUB mode.",
                PIPER_VOICE_MODEL_PATH,
            )
            self._loaded = True
            return
        logger.info("Loading Piper voice from %s ...", PIPER_VOICE_MODEL_PATH)
        self._voice = PiperVoice.load(str(PIPER_VOICE_MODEL_PATH), str(PIPER_VOICE_CONFIG_PATH))
        self._loaded = True

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Yields raw PCM audio bytes sentence-by-sentence for low-latency streaming playback.

        piper-tts >=1.3 (the piper1-gpl rewrite) replaced the old
        synthesize_stream_raw() method with synthesize(), which yields
        AudioChunk objects instead of raw bytes directly - unwrap them here.
        """
        self.load()
        if self._voice is None:
            return  # STUB mode: no audio to yield.

        for sentence in split_sentences(text):
            if not sentence:
                continue
            audio_chunks = await asyncio.to_thread(lambda: list(self._voice.synthesize(sentence)))
            for audio_chunk in audio_chunks:
                yield audio_chunk.audio_int16_bytes

    def list_available_voices(self) -> list:
        """Helper for the web UI voice-picker (page 5: 'preset library of voices')."""
        voices_dir = PIPER_VOICE_MODEL_PATH.parent
        if not voices_dir.exists():
            return []
        return sorted(p.name for p in voices_dir.glob("*.onnx"))


tts_engine = TTSEngine()