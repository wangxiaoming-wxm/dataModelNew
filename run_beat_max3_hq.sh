#!/usr/bin/env bash
# High-quality continuation AFTER P1 ord_noxb_new16.
# User mandate: do NOT downgrade params for speed. Time is acceptable.
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
mkdir -p logs/beat_max3 artifacts/beat_max3/train submissions

echo "[HQ] waiting for P1 ord_noxb_new16.npz ..."
while [[ ! -f artifacts/beat_max3/ord_noxb_new16.npz ]]; do
  # also accept 16 part files and bag them ourselves
  n=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
  if [[ "$n" -ge 16 ]]; then
    python3 - <<'PY'
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
parts=sorted(Path('artifacts/beat_max3/train').glob('part_ord_noxb_new16_s*.npz'))
assert len(parts)>=16
y=pd.read_csv('data/train.csv')['label'].astype(int).values
oofs=[np.load(p)['oof'] for p in parts[:16]]
tes=[np.load(p)['test_pred'] for p in parts[:16]]
aucs=[float(np.load(p)['auc']) for p in parts[:16]]
seeds=[int(p.stem.split('_s')[1]) for p in parts[:16]]
o=np.mean(oofs,0); t=np.mean(tes,0)
np.savez('artifacts/beat_max3/ord_noxb_new16.npz', oof=o, test_pred=t, per_seed=np.array(aucs), seeds=np.array(seeds), pool='mean')
meta={"tag":"ord_noxb_new16","pooled_oof_auc":float(roc_auc_score(y,o)),"per_seed":aucs,"seeds":seeds}
Path('artifacts/beat_max3/meta_ord_noxb_new16.json').write_text(json.dumps(meta,indent=2))
print(meta)
PY
    break
  fi
  echo "[HQ] wait parts=$n/16 $(date -Is)"; sleep 60
done
echo "[HQ] P1 ready"

python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_noxbnew --extra plus_strong noxb10 cat_w12_d5 ord_noxb_new16 || true
python3 src_beat/supervise.py --tag max3_stage_noxb16 --extra plus_strong noxb10 cat_w12_d5 ord_noxb_new16 || true

# -------- P2 HQ: deeper / more trees / 16 seeds each --------
echo "[HQ-P2] ord_noxb_d8x16 depth=8 iter=2000 lr=0.02"
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_d8x16 \
  --seeds 2300 2301 2302 2303 2304 2305 2306 2307 2308 2309 2310 2311 2312 2313 2314 2315 \
  --depth 8 --iterations 2000 --lr 0.02 --l2 12 --od-wait 200 \
  2>&1 | tee logs/beat_max3/ord_noxb_d8x16.log
python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_d8 --extra plus_strong noxb10 cat_w12_d5 ord_noxb_d8x16 || true
python3 src_beat/supervise.py --tag max3_full_d8 --extra plus_strong noxb10 cat_w12_d5 ord_noxb_new16 ord_noxb_d8x16 || true

echo "[HQ-P2] ord_noxb_slow7 depth=7 iter=2500 lr=0.015 l2=20"
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_slow7 \
  --seeds 2400 2401 2402 2403 2404 2405 2406 2407 2408 2409 2410 2411 2412 2413 2414 2415 \
  --depth 7 --iterations 2500 --lr 0.015 --l2 20 --od-wait 250 \
  2>&1 | tee logs/beat_max3/ord_noxb_slow7.log
python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_slow7 --extra plus_strong noxb10 cat_w12_d5 ord_noxb_slow7 || true
python3 src_beat/supervise.py --tag max3_full_slow --extra plus_strong noxb10 cat_w12_d5 ord_noxb_new16 ord_noxb_slow7 || true

echo "[HQ-P3] ord_noxb_b1x16 view=b1 depth=7 iter=2000"
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_b1x16 \
  --seeds 2500 2501 2502 2503 2504 2505 2506 2507 2508 2509 2510 2511 2512 2513 2514 2515 \
  --view b1 --depth 7 --iterations 2000 --lr 0.02 --l2 10 --od-wait 200 \
  2>&1 | tee logs/beat_max3/ord_noxb_b1x16.log
python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_b1 --extra plus_strong noxb10 cat_w12_d5 ord_noxb_b1x16 || true

# -------- P4 HQ plus: H3-ish, 8 seeds, 10-fold --------
echo "[HQ-P4] plus_hq8 (depth7 iter2500, 5fold x8 — train_plus)"
python3 src_beat/train_plus.py --tag plus_hq8 \
  --seeds 2600 2601 2602 2603 2604 2605 2606 2607 \
  --folds 5 --depth 7 --lr 0.02 \
  2>&1 | tee logs/beat_max3/plus_hq8.log
python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_plushq --extra plus_strong noxb10 cat_w12_d5 plus_hq8 || true
python3 src_beat/supervise.py --tag max3_dualplus_hq --extra plus_strong plus_hq8 cat_w12_d5 || true

# -------- P5 HQ methodology --------
echo "[HQ-P5] cofeh_hq8 B5+CoFEH ops Ordered"
python3 src_beat/train_method_arm.py --mode cofeh --tag cofeh_hq8 \
  --seeds 2700 2701 2702 2703 2704 2705 2706 2707 \
  2>&1 | tee logs/beat_max3/cofeh_hq8.log
python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_cofeh --extra plus_strong noxb10 cat_w12_d5 cofeh_hq8 || true

echo "[HQ-P5] goldmine_hq8 B5+GoldenFeatures"
python3 src_beat/train_method_arm.py --mode goldmine --tag goldmine_hq8 \
  --seeds 2800 2801 2802 2803 2804 2805 2806 2807 \
  2>&1 | tee logs/beat_max3/goldmine_hq8.log
python3 src_beat/refresh_stage_best.py || true
python3 src_beat/supervise.py --tag max3_best_gold --extra plus_strong noxb10 cat_w12_d5 goldmine_hq8 || true
python3 src_beat/supervise.py --tag max3_kitchen --extra plus_strong noxb10 cat_w12_d5 ord_noxb_new16 plus_hq8 cofeh_hq8 || true

python3 src_beat/refresh_stage_best.py || true
python3 - <<'PY'
import json
from pathlib import Path
rows=[]
for p in Path('artifacts/beat_max3').glob('report_*.json'):
    r=json.loads(p.read_text())
    rows.append((r['delta'], r['passed'], r['tag'], r['cand_nested'], r['spearman_vs_max3']))
rows.sort(reverse=True)
print('=== HQ FINAL LEADERBOARD ===')
for d,ok,tag,nest,sp in rows[:20]:
    print(f"{'PASS' if ok else 'FAIL'} Δ={d:+.5f} nest={nest:.5f} sp={sp:.4f} {tag}")
PY
echo HQ_LOOP_DONE
