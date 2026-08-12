#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"
if [[ "${1:-}" == "--verify" ]]; then
  exec "$PYTHON_BIN" -u src_super/fuse_fp_v4.py --verify-only
fi
exec "$PYTHON_BIN" -u src_super/fuse_fp_v4.py "$@"
