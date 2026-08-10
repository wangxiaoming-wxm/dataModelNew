#!/usr/bin/env python3
"""Split-half selection test: how much of a scan-selected 'best delta vs max3'
survives on rows that were not used to pick the recipe?

Candidate pool here is only ~62 recipes; the delivered scanner enumerated
combinations of size 2..5 over ~19 arms x 2 rules (thousands). So the shrinkage
measured here is a LOWER BOUND on the real selection optimism.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

OPUS = Path("/tmp/cmp_opus/20260810-cursor-opus5-4个")
V4 = Path("/tmp/cmp_v4/tree")
Y = pd.read_csv(V4 / "data/train.csv")["label"].astype(int).values
N = len(Y)

ARMS = {
    "merger_ord8": V4 / "artifacts/v4max3/merger_ord8.npz",
    "v2_cat_alt8": V4 / "artifacts/v4max3/v2_cat_alt8.npz",
    "ord_noxb_bag": V4 / "artifacts/v4max3/ord_noxb_bag.npz",
    "plus_strong": V4 / "artifacts/v4max3pro/plus_strong.npz",
    "noxb10": V4 / "artifacts/v4max3pro/noxb10.npz",
    "arm_gap": OPUS / "v5_honest/artifacts/arm_gap.npz",
    "v6_b5v2_8raw": OPUS / "v6_zcode/artifacts/v6_b5v2_8raw.npz",
    "plus_v10": V4 / "artifacts/v4max3/plus_v10.npz",
}
O = {}
for k, p in ARMS.items():
    d = np.load(p, allow_pickle=True)
    O[k] = rankdata(np.asarray(d["oof"], float)) / N

BASE = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
EXTRA = ["plus_strong", "noxb10", "arm_gap", "v6_b5v2_8raw", "plus_v10"]

cands = {}
for r in range(0, len(EXTRA) + 1):
    for combo in combinations(EXTRA, r):
        mem = BASE + list(combo)
        stack = np.vstack([O[m] for m in mem])
        cands["max|" + "+".join(combo) if combo else "max|<max3>"] = stack.max(0)
        if combo:
            cands["rmean|" + "+".join(combo)] = rankdata(stack.mean(0)) / N
base_key = "max|<max3>"
print(f"candidate pool size = {len(cands)}")


def auc_sub(v, idx):
    return roc_auc_score(Y[idx], v[idx])


rng = np.random.default_rng(7)
R = 200
sel_a, oos_b, best_b, ship_b = [], [], [], []
picked = {}
ship_key = "max|plus_strong+noxb10"
for _ in range(R):
    perm = rng.permutation(N)
    a, b = perm[: N // 2], perm[N // 2 :]
    ya, yb = Y[a], Y[b]
    if ya.sum() < 50 or yb.sum() < 50:
        continue
    base_a, base_b = auc_sub(cands[base_key], a), auc_sub(cands[base_key], b)
    da = {k: auc_sub(v, a) - base_a for k, v in cands.items()}
    db = {k: auc_sub(v, b) - base_b for k, v in cands.items()}
    win = max(da, key=da.get)
    picked[win] = picked.get(win, 0) + 1
    sel_a.append(da[win])
    oos_b.append(db[win])
    best_b.append(max(db.values()))
    ship_b.append(db[ship_key])

sel_a, oos_b, ship_b = np.array(sel_a), np.array(oos_b), np.array(ship_b)
print(f"\nrepeats={len(sel_a)}")
print(f"in-selection delta of the winner  : mean={sel_a.mean():+.5f}  sd={sel_a.std():.5f}")
print(f"same recipe on held-out half      : mean={oos_b.mean():+.5f}  sd={oos_b.std():.5f}")
print(f"shrinkage factor (oos/insel)      : {oos_b.mean()/sel_a.mean():.2f}")
print(f"held-out delta<=0 fraction        : {(oos_b <= 0).mean():.3f}")
print(f"\nshipped recipe (max|plus_strong+noxb10) on random halves:")
print(f"  mean={ship_b.mean():+.5f} sd={ship_b.std():.5f}  P(delta>0)={(ship_b>0).mean():.3f}")
print("\nwhich recipe wins the in-selection half (top 8):")
for k, c in sorted(picked.items(), key=lambda x: -x[1])[:8]:
    print(f"  {c/len(sel_a):6.1%}  {k}")
