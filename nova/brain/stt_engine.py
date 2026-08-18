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
import wave

from nova.config import WHISPER_MODEL_PATH, WHISPER_FILLER_PROMPT

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
# WHISPER_MODEL_PATH on first use, then cached there for subsequent runs).
# Options: tiny.en, base.en, small.en, medium.en, distil-large-v3, etc.
WHISPER_MODEL_SIZE = "base.en"


class STTEngine:
    def __init__(self):
        self._model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not _WHISPER_AVAILABLE:
            logger.warning(
                "faster-whisper not installed - STT engine running in STUB mode."
            )
            self._loaded = True
            return

        WHISPER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading faster-whisper model '%s' (cache dir: %s) ...",
            WHISPER_MODEL_SIZE, WHISPER_MODEL_PATH,
        )
        try:
            # int8 compute type keeps this fast and light on a Ryzen 3 CPU.
            self._model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                download_root=str(WHISPER_MODEL_PATH),
            )
        except Exception:
            logger.exception("Failed to load faster-whisper model - STT engine running in STUB mode.")
            self._model = None
        self._loaded = True

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