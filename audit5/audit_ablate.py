#!/usr/bin/env python3
"""Ablation: decompose the v4max3pro nested gain, and measure how much each
added arm moves the TEST vector relative to the LB-anchored max3 submission."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

OPUS = Path("/tmp/cmp_opus/20260810-cursor-opus5-4个")
V4 = Path("/tmp/cmp_v4/tree")
Y = pd.read_csv(V4 / "data/train.csv")["label"].astype(int).values
N = len(Y)
ANCHOR = pd.read_csv(V4 / "submissions/submission_v4_max3.csv")["label"].values

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
O, T = {}, {}
for k, p in ARMS.items():
    d = np.load(p, allow_pickle=True)
    key = "test_pred" if "test_pred" in d.files else "test"
    O[k] = rankdata(np.asarray(d["oof"], float)) / N
    T[k] = rankdata(np.asarray(d[key], float)) / len(d[key])


def nested(o, nb=5):
    out = np.zeros(N)
    for b in np.array_split(np.arange(N), nb):
        out[b] = rankdata(o[b]) / len(b)
    return float(roc_auc_score(Y, out))


BASE = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
base_o = np.maximum.reduce([O[m] for m in BASE])
base_n = nested(base_o)
print(f"max3 base nested = {base_n:.5f}\n")

print("== add-one / add-two ablations on top of max3 ==")
extras = ["plus_strong", "noxb10", "arm_gap", "v6_b5v2_8raw", "plus_v10"]
rows = []
for r in (1, 2):
    for combo in combinations(extras, r):
        o = np.maximum.reduce([base_o] + [O[m] for m in combo])
        t = np.maximum.reduce([np.maximum.reduce([T[m] for m in BASE])] + [T[m] for m in combo])
        lab = np.clip(t, 0.001, 0.999)
        sp = float(spearmanr(ANCHOR, lab).correlation)
        rows.append((("+".join(combo)), nested(o), nested(o) - base_n, sp))
for name, n, d, sp in sorted(rows, key=lambda x: -x[1]):
    star = "  <-- shipped v4max3pro" if name == "plus_strong+noxb10" else ""
    print(f"  max3+{name:26s} nested={n:.5f} delta={d:+.5f} sp_vs_anchor={sp:.5f}{star}")

print("\n== drop-one from the shipped 5-arm recipe ==")
FULL = BASE + ["plus_strong", "noxb10"]
full_o = np.maximum.reduce([O[m] for m in FULL])
print(f"  full(5 arms)                 nested={nested(full_o):.5f}")
for m in FULL:
    keep = [x for x in FULL if x != m]
    o = np.maximum.reduce([O[x] for x in keep])
    print(f"  drop {m:16s} nested={nested(o):.5f}  (contribution {nested(full_o)-nested(o):+.5f})")

print("\n== how far each shipped submission moves vs the 0.71222 anchor ==")
subs = {
    "v4_honest": OPUS / "v4_honest_zcode/submissions/submission_v4_honest.csv",
    "v5_honest": OPUS / "v5_honest/submissions/submission_v5_honest.csv",
    "v6_final": OPUS / "v6_zcode/submissions/submission_v6_final.csv",
    "v4max3pro": V4 / "submissions/submission_v4max3pro.csv",
}
ra = rankdata(ANCHOR)
n_te = len(ANCHOR)
for k, p in subs.items():
    v = pd.read_csv(p)["label"].values
    rv = rankdata(v)
    shift = np.abs(ra - rv)
    # rows crossing the top-decile boundary either way
    thr = n_te * 0.9
    in_a = ra > thr
    in_v = rv > thr
    churn = float(np.mean(in_a != in_v))
    print(
        f"  {k:10s} sp={spearmanr(ANCHOR, v).correlation:.5f} "
        f"mean|drank|={shift.mean():7.1f} max|drank|={shift.max():5.0f} "
        f"top10%_churn={churn:.4f} ties={int(n_te-len(np.unique(v)))}"
    )
