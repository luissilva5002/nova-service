#!/usr/bin/env bash
# scripts/download_models.sh
# Downloads the recommended model weights into ./models so they can be
# volume-mounted into the container (never baked into the Docker image).
# Run this from the project root on macOS, Linux, or Windows via WSL/Git Bash.
#
# Usage: bash scripts/download_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODELS_DIR/brain" "$MODELS_DIR/whisper" "$MODELS_DIR/piper" "$MODELS_DIR/minilm" "$MODELS_DIR/hf_cache"

download() {
  local destination="$1"
  local url="$2"
  local partial="$destination.part.$$"

  if [[ -s "$destination" ]]; then
    echo "==> Already present: ${destination##*/}"
    return
  fi

  echo "==> Downloading ${destination##*/}"
  if ! curl --fail --location --retry 3 --output "$partial" "$url"; then
    rm -f "$partial"
    echo "Download failed: $url" >&2
    return 1
  fi
  if [[ ! -s "$partial" ]]; then
    rm -f "$partial"
    echo "Download produced an empty file: $url" >&2
    return 1
  fi
  mv -f "$partial" "$destination"
}

echo "==> The Brain: Qwen 2.5 3B Instruct (Q4_K_M GGUF, ~2.2GB)"
download "$MODELS_DIR/brain/qwen2.5-3b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

echo "==> The Ears: whisper.cpp base.en model (~150MB)"
download "$MODELS_DIR/whisper/ggml-base.en.bin" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"

echo "==> The Mouth: Piper voice en_US-lessac-medium (~60MB)"
download "$MODELS_DIR/piper/en_US-lessac-medium.onnx" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
download "$MODELS_DIR/piper/en_US-lessac-medium.onnx.json" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

echo "==> MiniLM and faster-whisper assets are downloaded on first start when NOVA_OFFLINE=0."
echo "    The whisper.cpp file above is not used by faster-whisper."

echo "==> Done. Models placed under $MODELS_DIR"
