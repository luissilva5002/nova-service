#!/usr/bin/env bash
# scripts/download_models.sh
# Downloads the recommended model weights into ./models so they can be
# volume-mounted into the container (never baked into the Docker image).
# Run this from the project root on Windows (via WSL/Git-Bash) or Linux.
#
# Usage: bash scripts/download_models.sh

set -e

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR/brain" "$MODELS_DIR/whisper" "$MODELS_DIR/piper"

echo "==> The Brain: Qwen 2.5 3B Instruct (Q4_K_M GGUF, ~2.2GB)"
curl -L -o "$MODELS_DIR/brain/qwen2.5-3b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

echo "==> The Ears: whisper.cpp base.en model (~150MB)"
curl -L -o "$MODELS_DIR/whisper/ggml-base.en.bin" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"

echo "==> The Mouth: Piper voice en_US-lessac-medium (~60MB)"
curl -L -o "$MODELS_DIR/piper/en_US-lessac-medium.onnx" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
curl -L -o "$MODELS_DIR/piper/en_US-lessac-medium.onnx.json" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

echo "==> Done. Models placed under $MODELS_DIR"
echo "    Swap qwen2.5-3b-instruct-q4_k_m.gguf for a Llama 3.2 3B GGUF if you"
echo "    prefer that brain - just update NOVA_LLM_MODEL_PATH in nova/config.py"
echo "    or the corresponding environment variable."
