"""
nova/config.py
Central configuration for NOVA. Every path/tunable referenced by the rest
of the codebase lives here so the whole system can be reconfigured from
one place (or via environment variables / .env when running in Docker).
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Base paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # /app inside the container

MODELS_DIR = Path(os.getenv("NOVA_MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", BASE_DIR / "data"))
WEB_UI_DIR = Path(os.getenv("NOVA_WEB_UI_DIR", BASE_DIR / "web_ui"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "vector_store").mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Server settings
# --------------------------------------------------------------------------
HOST = os.getenv("NOVA_HOST", "0.0.0.0")
PORT = int(os.getenv("NOVA_PORT", "8000"))

# --------------------------------------------------------------------------
# The Brain - selectable local GGUF models (Q4_K_M)
# --------------------------------------------------------------------------
LLM_MODELS = {
    "qwen3-1.7b": {
        "label": "Qwen3 1.7B",
        "path": MODELS_DIR / "brain" / "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
    },
    "qwen3-0.6b": {
        "label": "Qwen3 0.6B",
        "path": MODELS_DIR / "brain" / "Qwen3-0.6B-Q4_K_M.gguf",
    },
    "qwen2.5-3b": {
        "label": "Qwen2.5 3B",
        "path": MODELS_DIR / "brain" / "qwen2.5-3b-instruct-q4_k_m.gguf",
    },
}
# Determine default LLM model. Prefer an explicit value set in this config file
# (hard-coded default) but still allow an environment override for advanced users.
_hardcoded_default = "qwen3-1.7b"
_requested_llm = os.getenv("NOVA_LLM_MODEL")
if _requested_llm and _requested_llm in LLM_MODELS:
    LLM_DEFAULT_MODEL_ID = _requested_llm
else:
    if _requested_llm and _requested_llm not in LLM_MODELS:
        import logging as _logging

        _logging.getLogger("nova.config").warning(
            "Requested NOVA_LLM_MODEL=%r not found in LLM_MODELS; falling back to hard-coded default %r",
            _requested_llm,
            _hardcoded_default,
        )
    LLM_DEFAULT_MODEL_ID = _hardcoded_default

# Backward-compatible override for deployments that set NOVA_LLM_MODEL_PATH.
LLM_MODEL_PATH = Path(os.getenv("NOVA_LLM_MODEL_PATH", LLM_MODELS[LLM_DEFAULT_MODEL_ID]["path"]))
LLM_CONTEXT_SIZE = int(os.getenv("NOVA_LLM_CTX", "4096"))
LLM_THREADS = int(os.getenv("OMP_NUM_THREADS", os.getenv("NOVA_LLM_THREADS", "4")))
LLM_GPU_LAYERS = int(os.getenv("NOVA_LLM_GPU_LAYERS", "0"))  # 0 = CPU only (Ryzen 3 target)

# System prompt persona - Tier 1 static memory (KV-cached, ~200 tokens target)
NOVA_PERSONA_PROMPT = os.getenv(
    "NOVA_PERSONA_PROMPT",
    "You are NOVA, a direct, witty personal AI assistant running locally on the "
    "user's own hardware. You control the user's devices and projects through "
    "registered tools. Be concise, natural, and only call a tool when the "
    "user's request clearly and explicitly requires it.\n\n"
    "Important: Only use the YouTube Music control tool when the user explicitly asks to play, pause, skip, stop, or otherwise control music playback (for example: 'Play <song/artist>', 'Pause the music', 'Skip this track'). Do NOT call the music tool for casual mentions of songs, notes, or unrelated conversation. If unsure, ask a clarifying question instead of invoking a tool.",
)

# --------------------------------------------------------------------------
# The Ears - faster-whisper (STT)
# --------------------------------------------------------------------------
# faster-whisper downloads a CTranslate2 model directory from Hugging Face.
# This must be a directory, not the legacy whisper.cpp .bin model file.
WHISPER_MODEL_CACHE_DIR = Path(
    os.getenv("NOVA_WHISPER_CACHE_DIR", MODELS_DIR / "whisper" / "faster_whisper_cache")
)
WHISPER_FILLER_PROMPT = (
    "Clean transcript without filler words, ums, or hesitations."
)

# --------------------------------------------------------------------------
# The Mouth - Piper TTS
# --------------------------------------------------------------------------
PIPER_VOICE_MODEL_PATH = Path(
    os.getenv("NOVA_PIPER_VOICE_PATH", MODELS_DIR / "piper" / "en_US-lessac-medium.onnx")
)
PIPER_VOICE_CONFIG_PATH = Path(
    os.getenv("NOVA_PIPER_VOICE_CONFIG", MODELS_DIR / "piper" / "en_US-lessac-medium.onnx.json")
)

# --------------------------------------------------------------------------
# Operational state (SQLite): conversation log, ingestion caches.
# NOT the persistent-memory "brain" - see PERSISTENT_MEMORY_VAULT_DIR below
# for that. Also Tier 3: ChromaDB vector store for ingested project code.
# --------------------------------------------------------------------------
CORE_STORE_DB_PATH = Path(os.getenv("NOVA_CORE_STORE_DB", DATA_DIR / "core_store.db"))
VECTOR_STORE_DIR = Path(os.getenv("NOVA_VECTOR_STORE_DIR", DATA_DIR / "vector_store"))
VECTOR_STORE_COLLECTION = os.getenv("NOVA_VECTOR_COLLECTION", "nova_projects")

# How many top-matching chunks the RAG retrieval step injects per turn.
# Kept low deliberately (see architecture doc: "Strict RAG Limits").
RAG_TOP_K = int(os.getenv("NOVA_RAG_TOP_K", "2"))

# --------------------------------------------------------------------------
# Persistent Memory - Obsidian-compatible Markdown vault (long-term memory)
# --------------------------------------------------------------------------
# Top-level (not nested under DATA_DIR) so it's easy to mount as its own
# Docker volume and browse/edit directly with Obsidian on the host machine.
PERSISTENT_MEMORY_VAULT_DIR = Path(os.getenv("NOVA_VAULT_DIR", BASE_DIR / "persistent_memory"))
(PERSISTENT_MEMORY_VAULT_DIR / "preferences").mkdir(parents=True, exist_ok=True)
(PERSISTENT_MEMORY_VAULT_DIR / "knowledge").mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Ingestion (dynamic project ingestion / two-tier filtering)
# --------------------------------------------------------------------------
DEFAULT_IGNORE_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".apk", ".so",
    ".zip", ".lock", ".exe", ".dll", ".class", ".jar", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".mp3", ".mp4", ".wav",
]
DEFAULT_IGNORE_DIR_HINTS = [
    "build", ".dart_tool", ".gradle", "Pods", ".git", "node_modules",
    "__pycache__", ".venv", "venv", "dist", ".idea", ".vs",
]

# --------------------------------------------------------------------------
# Host project mount (read-only) - where Flutter/other project folders live
# --------------------------------------------------------------------------
HOST_PROJECTS_DIR = Path(os.getenv("NOVA_HOST_PROJECTS_DIR", "/app/host_projects"))

# --------------------------------------------------------------------------
# Intent Classifier - trained MiniLM-embedding + LogisticRegression model
# (see classifier/train_intent_classifier.py). This is the PRIMARY intent
# detection mechanism; the Qwen3 LLM classification pass is only used as
# a fallback when this classifier's confidence is below threshold.
# --------------------------------------------------------------------------
CLASSIFIER_DIR = Path(os.getenv("NOVA_CLASSIFIER_DIR", BASE_DIR / "classifier"))
INTENT_CLASSIFIER_PATH = Path(
    os.getenv("NOVA_INTENT_CLASSIFIER_PATH", CLASSIFIER_DIR / "intent_classifier.joblib")
)
# Pick this from classifier/training_report.txt's threshold sweep - the
# row where acc_on_kept looks solid without forcing too many turns to chat.
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("NOVA_INTENT_CONFIDENCE_THRESHOLD", "0.6"))