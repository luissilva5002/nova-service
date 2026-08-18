#!/usr/bin/env bash
# Build, start, and follow the NOVA Docker service.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_DIR"

docker compose build
docker compose up -d
docker compose logs -f
