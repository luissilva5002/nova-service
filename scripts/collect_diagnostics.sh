#!/usr/bin/env bash
set -euo pipefail
out="diagnostics_$(date +%Y%m%d_%H%M%S).txt"
{
  uname -a
  lscpu | grep -E 'Model name|CPU\(s\)|Core\(s\)'
  free -h
  docker compose config -q && echo "compose=ok" || echo "compose=failed"
  docker compose ps
  docker compose exec -T nova_server sh -c 'env | grep -E "^(NOVA_|HF_|TRANSFORMERS_|ANONYMIZED_)" | sed -E "s/(KEY|TOKEN|SECRET|PASSWORD)=.*/\1=<redacted>/"' || true
  docker compose exec -T nova_server pip show llama-cpp-python sentence-transformers faster-whisper chromadb || true
  ls -lh models/ || true
  docker compose logs --tail=500 2>&1 | grep -ci huggingface || true
  docker compose logs --tail=100 2>&1 | grep -E 'Classified|LLM decision|prompt_tokens|ttft|offline=' || true
  vmstat 1 5
  docker stats --no-stream || true
} 2>&1 | sed -E 's/(KEY|TOKEN|SECRET|PASSWORD)[=:][^ ]+/\1=<redacted>/g' > "$out"
echo "wrote $out"
