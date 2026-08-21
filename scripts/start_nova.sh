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

docker compose build
docker compose up -d
docker compose logs -f
