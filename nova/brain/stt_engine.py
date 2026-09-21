"""
nova/brain/stt_engine.py

The Ears - faster-whisper (CTranslate2-based Whisper inference). NOT an
LLM: a small acoustic transcription model. Uses prompt-conditioning
(initial_prompt) to strip filler words ("uh", "um", hesitations) directly
during decoding, instead of running a second LLM pass to clean the
transcript (see architecture doc "Latency Breakdown" - this is the
~2.5s optimization).

Swapped from whisper-cpp-python to faster-whisper: it's pure pip-install
(no CMake/C++ compile step, which was failing in Docker with modern
CMake versions), actively maintained, and runs efficiently on CPU with
int8 quantization.
"""
import io
import logging
import re
import subprocess
import wave

from nova.config import WHISPER_MODEL_CACHE_DIR, WHISPER_MODEL_DIR, WHISPER_FILLER_PROMPT, NOVA_OFFLINE

logger = logging.getLogger("nova.stt_engine")

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WHISPER_AVAILABLE = False

# Regex fallback filler-word strip, used in STUB mode or as a safety net
# even when Whisper's own prompt conditioning already did most of the work.
_FILLER_PATTERN = re.compile(r"\b(uh+|um+|erm+|like|you know)\b[,]?\s*", re.IGNORECASE)

# faster-whisper model size/name (auto-downloaded from Hugging Face into
# WHISPER_MODEL_CACHE_DIR on first use, then cached there for subsequent runs).
# Options: tiny.en, base.en, small.en, medium.en, distil-large-v3, etc.
WHISPER_MODEL_SIZE = "base.en"


class STTEngine:
    def __init__(self):
        self._model = None
        self._loaded = False
        self._load_attempted = False

    def load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not _WHISPER_AVAILABLE:
            logger.warning(
                "faster-whisper not installed - STT engine running in STUB mode."
            )
            return

        WHISPER_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading faster-whisper model '%s' (cache dir: %s) ...",
            WHISPER_MODEL_SIZE, WHISPER_MODEL_CACHE_DIR,
        )
        try:
            # int8 compute type keeps this fast and light on a Ryzen 3 CPU.
            model_source = str(WHISPER_MODEL_DIR) if WHISPER_MODEL_DIR.exists() else WHISPER_MODEL_SIZE
            if NOVA_OFFLINE and not WHISPER_MODEL_DIR.exists():
                raise FileNotFoundError(f"Offline STT model directory is missing: {WHISPER_MODEL_DIR}")
            self._model = WhisperModel(
                model_source,
                device="cpu",
                compute_type="int8",
                download_root=str(WHISPER_MODEL_CACHE_DIR),
                local_files_only=NOVA_OFFLINE,
            )
        except Exception:
            logger.exception("Failed to load faster-whisper model - STT engine running in STUB mode.")
            self._model = None
        self._loaded = self._model is not None

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        audio_bytes: raw PCM16 mono audio (as streamed from the Flutter/web
        client over the WebSocket, after any format conversion). Returns
        filler-cleaned text.
        """
        self.load()
        if self._model is None:
            # STUB mode: nothing to transcribe without a real model.
            return ""
        if not audio_bytes:
            return ""

        wav_buffer = self._pcm16_to_wav(audio_bytes, sample_rate)
        segments, _info = self._model.transcribe(
            wav_buffer,
            initial_prompt=WHISPER_FILLER_PROMPT,
            language="en",
            vad_filter=True,  # skip silence, reduces hallucinated fillers further
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return self._strip_residual_fillers(text)

    def transcribe_media(self, audio_bytes: bytes) -> str:
        """Transcribe browser-recorded WebM/Opus audio via FFmpeg conversion."""
        if not audio_bytes:
            return ""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ac", "1", "-ar", "16000", "pipe:1",
                ],
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("Could not decode microphone audio: %s", exc)
            return ""
        return self.transcribe(result.stdout, sample_rate=16000)

    @staticmethod
    def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int) -> io.BytesIO:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        buffer.seek(0)
        return buffer

    @staticmethod
    def _strip_residual_fillers(text: str) -> str:
        cleaned = _FILLER_PATTERN.sub("", text)
        return re.sub(r"\s{2,}", " ", cleaned).strip()


stt_engine = STTEngine()
