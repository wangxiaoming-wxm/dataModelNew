#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1
mkdir -p logs/beat_max3

echo "[wait] for ord_noxb_new16.npz ..."
while [[ ! -f artifacts/beat_max3/ord_noxb_new16.npz ]]; do sleep 30; done
echo "[got] ord_noxb_new16 — supervise combos"
python3 src_beat/supervise.py --tag max3_best_noxbnew --extra plus_strong noxb10 cat_w12_d5 ord_noxb_new16 || true
python3 src_beat/supervise.py --tag max3_plus_noxbnew --extra plus_strong ord_noxb_new16 || true

echo "[P4] plus_new8 training"
python3 src_beat/train_plus.py --tag plus_new8 --seeds 2600 2601 2602 2603 2604 2605 2606 2607 \
  2>&1 | tee logs/beat_max3/plus_new8.log
python3 src_beat/supervise.py --tag max3_best_plusnew --extra plus_strong noxb10 cat_w12_d5 plus_new8 || true
python3 src_beat/supervise.py --tag max3_plusnew --extra plus_new8 || true
python3 src_beat/supervise.py --tag max3_dualplus --extra plus_strong plus_new8 cat_w12_d5 || true

# leaderboard dump
python3 - <<'PY'
import json
from pathlib import Path
rows=[]
for p in Path('artifacts/beat_max3').glob('report_*.json'):
    r=json.loads(p.read_text())
    rows.append((r['delta'], r['passed'], r['tag'], r['cand_nested'], r['spearman_vs_max3']))
rows.sort(reverse=True)
print('=== LEADERBOARD ===')
for d,ok,tag,nest,sp in rows:
    print(f"{'PASS' if ok else 'FAIL'} Δ={d:+.5f} nest={nest:.5f} sp={sp:.4f} {tag}")
PY
echo FOLLOWUP_DONE
