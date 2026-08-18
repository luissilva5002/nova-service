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
# The Brain (single LLM) - Qwen 2.5 3B or Llama 3.2 3B, GGUF, Q4_K_M
# --------------------------------------------------------------------------
LLM_MODEL_PATH = Path(
    os.getenv("NOVA_LLM_MODEL_PATH", MODELS_DIR / "brain" / "qwen2.5-3b-instruct-q4_k_m.gguf")
)
LLM_CONTEXT_SIZE = int(os.getenv("NOVA_LLM_CTX", "4096"))
LLM_THREADS = int(os.getenv("OMP_NUM_THREADS", os.getenv("NOVA_LLM_THREADS", "4")))
LLM_GPU_LAYERS = int(os.getenv("NOVA_LLM_GPU_LAYERS", "0"))  # 0 = CPU only (Ryzen 3 target)

# System prompt persona - Tier 1 static memory (KV-cached, ~200 tokens target)
NOVA_PERSONA_PROMPT = os.getenv(
    "NOVA_PERSONA_PROMPT",
    "You are NOVA, a direct, witty personal AI assistant running locally on the "
    "user's own hardware. You control the user's devices and projects through "
    "registered tools. Be concise, natural, and only call a tool when the "
    "user's request maps to one of the tool schemas provided to you.",
)

# --------------------------------------------------------------------------
# The Ears - whisper.cpp (STT)
# --------------------------------------------------------------------------
WHISPER_MODEL_PATH = Path(
    os.getenv("NOVA_WHISPER_MODEL_PATH", MODELS_DIR / "whisper" / "ggml-base.en.bin")
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
# Memory (Tier 2: SQLite core store, Tier 3: ChromaDB vector store)
# --------------------------------------------------------------------------
CORE_STORE_DB_PATH = Path(os.getenv("NOVA_CORE_STORE_DB", DATA_DIR / "core_store.db"))
VECTOR_STORE_DIR = Path(os.getenv("NOVA_VECTOR_STORE_DIR", DATA_DIR / "vector_store"))
VECTOR_STORE_COLLECTION = os.getenv("NOVA_VECTOR_COLLECTION", "nova_projects")

# How many top-matching chunks the RAG retrieval step injects per turn.
# Kept low deliberately (see architecture doc: "Strict RAG Limits").
RAG_TOP_K = int(os.getenv("NOVA_RAG_TOP_K", "2"))

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
