#!/usr/bin/env bash
# Download a selectable NOVA brain model without re-downloading STT/TTS assets.
# Usage: bash scripts/download_brain_models.sh [qwen2.5-3b|qwen3-1.7b|qwen3-0.6b|all]
set -euo pipefail

MODEL_ID="${1:-all}"
MODELS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/models/brain"
mkdir -p "$MODELS_DIR"

download() {
  local name="$1"
  local url="$2"
  local target="$MODELS_DIR/$name"
  local partial="$target.part"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    echo "==> Already present: $name"
    return
  fi
  rm -f "$partial"
  echo "==> Downloading $name"
  curl -fL --retry 3 -o "$partial" "$url"
  mv "$partial" "$target"
}

case "$MODEL_ID" in
  qwen2.5-3b)
    download "qwen2.5-3b-instruct-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    ;;
  qwen3-1.7b)
    download "Qwen_Qwen3-1.7B-Q4_K_M.gguf" "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    ;;
  qwen3-0.6b)
    download "Qwen3-0.6B-Q4_K_M.gguf" "https://huggingface.co/second-state/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
    ;;
  all)
    "$0" qwen2.5-3b
    "$0" qwen3-1.7b
    "$0" qwen3-0.6b
    ;;
  *)
    echo "Unknown model: $MODEL_ID" >&2
    exit 2
    ;;
esac
