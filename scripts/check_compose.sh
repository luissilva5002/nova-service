#!/usr/bin/env bash
set -euo pipefail
docker compose config -q
echo "docker compose configuration is valid"
