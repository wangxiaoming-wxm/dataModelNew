#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"
if [[ "${1:-}" == "--verify" ]]; then
  exec "$PYTHON_BIN" -u vz20/src/build_vz20.py --verify
fi
exec "$PYTHON_BIN" -u vz20/src/build_vz20.py "$@"
