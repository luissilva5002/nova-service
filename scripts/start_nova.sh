#!/usr/bin/env bash
# Build, start, and follow the NOVA Docker service.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI was not found in this WSL environment." >&2
  echo "On Windows, enable Docker Desktop's WSL integration and restart WSL." >&2
  echo "Then reopen this shell and re-run: bash scripts/start_nova.sh" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker is installed, but the Docker Compose plugin is unavailable." >&2
  echo "Please install/enable Docker Desktop's WSL integration or the Docker Compose plugin." >&2
  exit 1
fi

# Clean any stale compose state before starting; this avoids some Docker Desktop
# WSL bind-mount issues when a previous container was interrupted mid-start.
docker compose down --remove-orphans >/dev/null 2>&1 || true
docker compose rm -sf nova_core >/dev/null 2>&1 || true

docker compose build

if ! docker compose up -d; then
  echo "" >&2
  echo "Docker Desktop / WSL bind-mount startup failed." >&2
  echo "This usually means Docker Desktop is in a stale state." >&2
  echo "Fix it by:" >&2
  echo "  1) Fully close Docker Desktop." >&2
  echo "  2) In WSL: wsl --shutdown" >&2
  echo "  3) Re-open WSL and start Docker Desktop again." >&2
  echo "  4) Run: bash scripts/start_nova.sh" >&2
  exit 1
fi

docker compose logs -f
