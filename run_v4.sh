#!/usr/bin/env bash
# Reproduce submission_v4.csv from shipped arm artefacts (no retrain).
#
# What this does:
#   1. fuse4.py  — nested selection over 20 block seeds → submission_v4.csv
#   2. audit_v4.py — independent honesty gates + target gate (0.707)
#
# Expected headline (must match docs/V4.md):
#   nested_oof_mean ≈ 0.70303
#   submitted_rule  = views_max_10_20_r16_r16b
#   honesty_passed  = true
#   target_reached  = false
#
# Full retrain from seeds is multi-day on 4 cores; see docs/V4.md §复现.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src2:src3:src4

python3 -m pip install -q -r requirements.txt

mkdir -p artifacts/audit_v4 submissions

echo "=== fuse4 ==="
python3 src4/fuse4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4.csv \
  --report artifacts/v4/fusion_report_v4.json

echo "=== audit_v4 (target 0.707) ==="
set +e
python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4.csv \
  --target 0.707 \
  --out artifacts/audit_v4/audit.json
rc=$?
set -e

python3 - <<'PY'
import json
r = json.load(open("artifacts/v4/fusion_report_v4.json"))
a = json.load(open("artifacts/audit_v4/audit.json"))
print(f"nested_oof_mean = {r['nested_oof_mean']:.5f}")
print(f"submitted_rule  = {r['submitted_rule']}")
print(f"honesty_passed  = {a['honesty_passed']}")
print(f"target_reached  = {a['gates']['target_reached']}")
print(f"audit_exit      = {a.get('passed')}")
PY

# target gate is expected to fail; honesty must pass
python3 - <<'PY'
import json, sys
a = json.load(open("artifacts/audit_v4/audit.json"))
if not a["honesty_passed"]:
    sys.exit("honesty gates failed")
print("run_v4.sh OK (honesty pass; target 0.707 intentionally unmet)")
PY
