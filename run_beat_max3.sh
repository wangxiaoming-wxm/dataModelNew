#!/usr/bin/env bash
# Long-running beat-max3 training loop.
# Goal: produce new arms that raise max(rank) nested vs max3 under supervisor gates.
set -euo pipefail
cd /workspace
mkdir -p artifacts/beat_max3/train logs/beat_max3 submissions

export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# Copy curated extra arms into beat_max3 namespace
cp -n /tmp/audit_all/v4max3pro/artifacts/v4max3/plus_v10.npz artifacts/beat_max3/plus_v10.npz 2>/dev/null || true
cp -n /tmp/audit_all/v4max3pro/artifacts/v4max3/ordered_bag.npz artifacts/beat_max3/ordered_bag.npz 2>/dev/null || true
cp -n /tmp/audit_all/v4max3pro/artifacts/v4max3/mine_noxb8.npz artifacts/beat_max3/mine_noxb8.npz 2>/dev/null || true
cp -n /tmp/audit_all/v4max3pro/artifacts/v4max3/b7_closest.npz artifacts/beat_max3/b7_closest.npz 2>/dev/null || true
cp -n /tmp/audit_all/v4max3pro/artifacts/v4max3pronew/semantic_rmse.npz artifacts/beat_max3/semantic_rmse.npz 2>/dev/null || true
# normalize cat_w12 keys
python3 - <<'PY'
import numpy as np
from pathlib import Path
ART=Path('artifacts/beat_max3')
for src,name in [
 ('artifacts/v4_ext/arm_cat_w12_d5.npz','cat_w12_d5'),
 ('artifacts/v4_ext/arm_cat_w12_d6.npz','cat_w12_d6'),
 ('artifacts/v4_ext/arm_gap_v5.npz','gap_v5'),
]:
    p=Path(src)
    if not p.exists():
        continue
    d=np.load(p)
    te=d['test'] if 'test' in d.files else d['test_pred']
    np.savez(ART/f'{name}.npz', oof=d['oof'], test_pred=te, y=d['y'] if 'y' in d.files else None)
    print('normalized', name)
PY

# P0: fuse existing best recipes under supervisor
python3 src_beat/supervise.py --tag max3_plus --extra plus_strong || true
python3 src_beat/supervise.py --tag max3_pro --extra plus_strong noxb10 || true
python3 src_beat/supervise.py --tag max3_best --extra plus_strong noxb10 cat_w12_d5 || true
python3 src_beat/supervise.py --tag max3_plus_w12 --extra plus_strong cat_w12_d5 || true
python3 src_beat/supervise.py --tag max3_pro_sem --extra plus_strong noxb10 semantic_rmse || true

# P1: new ord_noxb seeds (champion protocol, ES allowed)
echo "[P1] training ord_noxb_new16 ..."
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_new16 \
  --seeds 2100 2101 2102 2103 2104 2105 2106 2107 2108 2109 2110 2111 2112 2113 2114 2115 \
  --depth 7 --iterations 1200 --lr 0.03 \
  2>&1 | tee logs/beat_max3/ord_noxb_new16.log

# Fuse: replace? NO — keep original bag AND add new bag as 4th ES arm
python3 src_beat/supervise.py --tag max3_noxbnew --extra plus_strong noxb10 ord_noxb_new16 || true
python3 src_beat/supervise.py --tag max3_plus_noxbnew --extra plus_strong ord_noxb_new16 || true

# P2: depth/lr variants of noxb family
echo "[P2] training ord_noxb_d6 ..."
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_d6 \
  --seeds 2200 2201 2202 2203 2204 2205 2206 2207 \
  --depth 6 --iterations 1400 --lr 0.03 \
  2>&1 | tee logs/beat_max3/ord_noxb_d6.log
python3 src_beat/supervise.py --tag max3_d6 --extra plus_strong noxb10 ord_noxb_d6 || true

echo "[P2] training ord_noxb_d8 ..."
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_d8 \
  --seeds 2300 2301 2302 2303 2304 2305 2306 2307 \
  --depth 8 --iterations 1000 --lr 0.025 \
  2>&1 | tee logs/beat_max3/ord_noxb_d8.log
python3 src_beat/supervise.py --tag max3_d8 --extra plus_strong noxb10 ord_noxb_d8 || true

echo "[P2] training ord_noxb_l2strong ..."
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_l2s \
  --seeds 2400 2401 2402 2403 2404 2405 2406 2407 \
  --depth 7 --iterations 1400 --lr 0.02 --l2 20 \
  2>&1 | tee logs/beat_max3/ord_noxb_l2s.log
python3 src_beat/supervise.py --tag max3_l2s --extra plus_strong noxb10 ord_noxb_l2s || true

# P3: b1 view ordered noxb
echo "[P3] training ord_noxb_b1 ..."
python3 src_beat/train_ord_noxb.py \
  --tag ord_noxb_b1 \
  --seeds 2500 2501 2502 2503 2504 2505 2506 2507 \
  --view b1 --depth 7 --iterations 1200 --lr 0.03 \
  2>&1 | tee logs/beat_max3/ord_noxb_b1.log
python3 src_beat/supervise.py --tag max3_b1 --extra plus_strong noxb10 ord_noxb_b1 || true

# Final leaderboard of local candidates
python3 - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path('artifacts/beat_max3').glob('report_*.json')):
    r=json.loads(p.read_text())
    rows.append((r.get('delta',-1), r.get('passed'), r.get('tag'), r.get('cand_nested'), r.get('spearman_vs_max3'), r.get('arms')))
rows.sort(reverse=True)
print('=== LOCAL CANDIDATE LEADERBOARD (by nested delta vs max3) ===')
for d,ok,tag,nest,sp,arms in rows:
    print(f"{'+' if ok else '-'} Δ={d:+.5f} nest={nest:.5f} sp={sp:.4f} {tag} arms={arms}")
best=[r for r in rows if r[1]]
if best:
    print('BEST_SHIP', best[0][2], 'delta', best[0][0])
PY

echo "LOOP_DONE"
