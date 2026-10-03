#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
temp_root="$(mktemp -d)"
trap 'rm -rf "$temp_root"' EXIT

mkdir -p "$temp_root/scripts" "$temp_root/bin"
cp "$repo_root/scripts/download_models.sh" "$temp_root/scripts/"

cat > "$temp_root/bin/curl" <<'MOCK_CURL'
#!/usr/bin/env bash
set -euo pipefail

output=""
while (($#)); do
  if [[ "$1" == "--output" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done

printf 'call\n' >> "$MOCK_CURL_CALLS"
printf 'partial fixture\n' > "$output"
if [[ "${MOCK_CURL_FAIL:-0}" == "1" ]]; then
  exit 22
fi
MOCK_CURL
chmod +x "$temp_root/bin/curl"

set +e
PATH="$temp_root/bin:$PATH" \
MOCK_CURL_CALLS="$temp_root/curl-calls" \
MOCK_CURL_FAIL=1 \
  bash "$temp_root/scripts/download_models.sh" >/dev/null 2>&1
download_status=$?
set -e

expected_model="$temp_root/models/brain/qwen2.5-3b-instruct-q4_k_m.gguf"
if [[ "$download_status" -eq 0 ]]; then
  echo "Expected the downloader to fail when curl fails." >&2
  exit 1
fi
if [[ -e "$expected_model" || -e "$expected_model.part."* ]]; then
  echo "A failed download left a partial model file behind." >&2
  exit 1
fi

PATH="$temp_root/bin:$PATH" \
MOCK_CURL_CALLS="$temp_root/curl-calls" \
  bash "$temp_root/scripts/download_models.sh" >/dev/null
if [[ ! -s "$expected_model" ]]; then
  echo "The downloader did not install a non-empty model file." >&2
  exit 1
fi
if [[ "$(wc -l < "$temp_root/curl-calls")" -ne 5 ]]; then
  echo "Expected one failed request followed by four successful downloads." >&2
  exit 1
fi

PATH="$temp_root/bin:$PATH" \
MOCK_CURL_CALLS="$temp_root/curl-calls" \
  bash "$temp_root/scripts/download_models.sh" >/dev/null
if [[ "$(wc -l < "$temp_root/curl-calls")" -ne 5 ]]; then
  echo "The downloader did not skip existing non-empty files." >&2
  exit 1
fi

echo "Downloader failure cleanup and idempotence checks passed."
