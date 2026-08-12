#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python3}" -u -m src_rebuild.cli "$@"
