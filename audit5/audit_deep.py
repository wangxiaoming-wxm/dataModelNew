#!/usr/bin/env python3
"""Deeper audit: arm collinearity, who-wins-the-max structure (OOF vs test),
paired bootstrap on the nested-AUC deltas, and label-shuffle sanity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
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
}
PROTO = {
    "merger_ord8": "honest 5f/800it/8seed",
    "v2_cat_alt8": "honest 5f/800it/8seed",
    "ord_noxb_bag": "ES 5f/8seed",
    "plus_strong": "ES 10f (plus family)",
    "noxb10": "ES 10f/8seed",
    "arm_gap": "honest 5f/500it/4seed",
    "v6_b5v2_8raw": "ES 5f/8seed raw-avg",
}

O, T = {}, {}
for k, p in ARMS.items():
    d = np.load(p, allow_pickle=True)
    key = "test_pred" if "test_pred" in d.files else "test"
    O[k] = rankdata(np.asarray(d["oof"], float)) / N
    T[k] = rankdata(np.asarray(d[key], float)) / len(d[key])

names = list(ARMS)
print("== single-arm OOF AUC / protocol ==")
for k in names:
    print(f"{k:14s} oof_auc={roc_auc_score(Y, O[k]):.5f}  {PROTO[k]}")

print("\n== pairwise SPEARMAN (upper=OOF, lower=TEST) ==")
hdr = "".join(f"{k[:10]:>12s}" for k in names)
print(f"{'':14s}{hdr}")
for i, a in enumerate(names):
    row = ""
    for j, b in enumerate(names):
        v = (
            spearmanr(O[a], O[b]).correlation
            if j >= i
            else spearmanr(T[a], T[b]).correlation
        )
        row += f"{v:12.4f}"
    print(f"{a:14s}{row}")

FUSIONS = {
    "v4_honest": ["merger_ord8", "v2_cat_alt8"],
    "v4_max3": ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"],
    "v5_honest": ["merger_ord8", "v2_cat_alt8", "arm_gap"],
    "v6_final": ["merger_ord8", "v2_cat_alt8", "v6_b5v2_8raw"],
    "v4max3pro": ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag", "plus_strong", "noxb10"],
}


def nested(oof, y, nb=5):
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), nb):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


print("\n== who wins the elementwise max (OOF split by label vs TEST) ==")
for f, mem in FUSIONS.items():
    Om = np.vstack([O[m] for m in mem])
    Tm = np.vstack([T[m] for m in mem])
    wo = np.argmax(Om, 0)
    wt = np.argmax(Tm, 0)
    parts = []
    for i, m in enumerate(mem):
        p_all = (wo == i).mean()
        p_pos = (wo[Y == 1] == i).mean()
        p_te = (wt == i).mean()
        parts.append(f"{m}: oof={p_all:.3f}(pos={p_pos:.3f}) test={p_te:.3f}")
    print(f"  {f:11s} " + " | ".join(parts))

print("\n== nested-AUC deltas vs v4_max3, paired bootstrap (2000 resamples) ==")
rng = np.random.default_rng(12345)
FO = {f: np.maximum.reduce([O[m] for m in mem]) for f, mem in FUSIONS.items()}
base = FO["v4_max3"]
B = 2000
idx_all = np.arange(N)
boots = {f: np.empty(B) for f in FUSIONS if f != "v4_max3"}
for b in range(B):
    idx = rng.choice(idx_all, N, replace=True)
    yb = Y[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    ab = roc_auc_score(yb, base[idx])
    for f in boots:
        boots[f][b] = roc_auc_score(yb, FO[f][idx]) - ab
summary = {}
for f in boots:
    d_point = nested(FO[f], Y) - nested(base, Y)
    lo, hi = np.percentile(boots[f], [2.5, 97.5])
    p_gt0 = float((boots[f] > 0).mean())
    summary[f] = dict(
        nested_delta=round(d_point, 5),
        boot_mean=round(float(boots[f].mean()), 5),
        ci95=[round(float(lo), 5), round(float(hi), 5)],
        p_delta_gt_0=p_gt0,
    )
    print(f"  {f:11s} nested_delta={d_point:+.5f}  boot95%CI=[{lo:+.5f},{hi:+.5f}]  P(delta>0)={p_gt0:.3f}")

print("\n== per-block nested AUC (5 contiguous blocks) ==")
for f, o in FO.items():
    per = []
    for blk in np.array_split(np.arange(N), 5):
        per.append(roc_auc_score(Y[blk], o[blk]))
    print(f"  {f:11s} " + " ".join(f"{v:.4f}" for v in per) + f"   mean={np.mean(per):.5f} sd={np.std(per):.4f}")

print("\n== label-shuffle sanity (should be ~0.5) ==")
rng2 = np.random.default_rng(0)
for f, o in FO.items():
    vals = [roc_auc_score(rng2.permutation(Y), o) for _ in range(20)]
    print(f"  {f:11s} mean={np.mean(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}")

Path("/tmp/audit/out").mkdir(parents=True, exist_ok=True)
Path("/tmp/audit/out/deep.json").write_text(json.dumps(summary, indent=2))
