"""Part 2: quantify optimism and selection risk of the V4max3pro recipe.

1. Incremental contribution of each new arm.
2. OOF-vs-test "max win-rate" asymmetry -> detects arms whose OOF scale is
   inflated relative to their test predictions (ES / 10-fold optimism).
3. Paired bootstrap CI of the nested-AUC delta vs max3.
4. max vs rank-mean aggregation, to see if `max` is an OOF-specific winner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace")
y = pd.read_csv(ROOT / "data" / "train.csv")["label"].astype(int).values
A3, AP = ROOT / "artifacts" / "v4max3", ROOT / "artifacts" / "v4max3pro"

P = {
    "merger_ord8": A3 / "merger_ord8.npz",
    "v2_cat_alt8": A3 / "v2_cat_alt8.npz",
    "ord_noxb_bag": A3 / "ord_noxb_bag.npz",
    "ordered_bag": A3 / "ordered_bag.npz",
    "plus_v10": A3 / "plus_v10.npz",
    "plus_strong": AP / "plus_strong.npz",
    "noxb10": AP / "noxb10.npz",
    "main10_ord_es": AP / "main10_ord_es.npz",
}
arms = {}
for k, p in P.items():
    d = np.load(p, allow_pickle=True)
    o = np.asarray(d["oof"], float)
    t = np.asarray(d["test_pred"] if "test_pred" in d else d["test"], float)
    arms[k] = {"oof": rankdata(o) / len(o), "te": rankdata(t) / len(t)}

MAX3 = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
RECIPE = MAX3 + ["plus_strong", "noxb10"]


def blocks(n=len(y), nb=5):
    return np.array_split(np.arange(n), nb)


BLK = blocks()


def nested(oof):
    out = np.zeros(len(oof))
    for b in BLK:
        out[b] = rankdata(oof[b]) / len(b)
    return out


def nauc(oof):
    return float(roc_auc_score(y, nested(oof)))


def rmax(ms, which):
    return np.vstack([arms[m][which] for m in ms]).max(axis=0)


base = nauc(rmax(MAX3, "oof"))
print(f"max3 nested = {base:.6f}\n")

print("=== 1. INCREMENTAL CONTRIBUTION (nested delta vs max3) ===")
for extra in (
    ["plus_strong"], ["noxb10"], ["ordered_bag"], ["plus_v10"], ["main10_ord_es"],
    ["plus_strong", "noxb10"],
):
    ms = MAX3 + extra
    d = nauc(rmax(ms, "oof")) - base
    sp = spearmanr(rmax(ms, "te"), rmax(MAX3, "te")).correlation
    print(f"  +{'+'.join(extra):26s} nested={nauc(rmax(ms,'oof')):.6f} delta={d:+.6f} sp_te={sp:.5f}")

print("\n=== 2. OOF vs TEST asymmetry of the max operator ===")
print("   (each arm is rank-uniform on its own side; if an arm wins the max on a")
print("    much larger share of OOF rows than TEST rows, its OOF is inflated)")
for side in ("oof", "te"):
    st = np.vstack([arms[m][side] for m in RECIPE])
    win = st.argmax(axis=0)
    sh = {RECIPE[i]: float((win == i).mean()) for i in range(len(RECIPE))}
    print(f"  {side:4s} win-share " + "  ".join(f"{k}={v:.4f}" for k, v in sh.items()))
st_o = np.vstack([arms[m]["oof"] for m in RECIPE]).argmax(axis=0)
st_t = np.vstack([arms[m]["te"] for m in RECIPE]).argmax(axis=0)
print("  delta(oof-test) win-share:")
for i, m in enumerate(RECIPE):
    print(f"    {m:14s} {(st_o==i).mean()-(st_t==i).mean():+.4f}")

print("\n  rank-displacement caused by the two new arms:")
print(f"    spearman(max3_oof, recipe_oof) = {spearmanr(rmax(MAX3,'oof'), rmax(RECIPE,'oof')).correlation:.6f}")
print(f"    spearman(max3_te , recipe_te ) = {spearmanr(rmax(MAX3,'te'),  rmax(RECIPE,'te')).correlation:.6f}")

print("\n  pairwise spearman of arms (oof / test):")
for a, b in [("ord_noxb_bag", "noxb10"), ("plus_strong", "plus_v10"),
             ("merger_ord8", "noxb10"), ("plus_strong", "ord_noxb_bag")]:
    print(f"    {a:14s} vs {b:14s} oof={spearmanr(arms[a]['oof'],arms[b]['oof']).correlation:.4f} "
          f"te={spearmanr(arms[a]['te'],arms[b]['te']).correlation:.4f}")

print("\n=== 3. PAIRED BOOTSTRAP of nested delta (recipe - max3) ===")
o_b, o_r = nested(rmax(MAX3, "oof")), nested(rmax(RECIPE, "oof"))
rng = np.random.default_rng(7)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
ds = []
for _ in range(2000):
    ip = rng.choice(pos, len(pos), replace=True)
    ineg = rng.choice(neg, len(neg), replace=True)
    idx = np.concatenate([ip, ineg])
    yy = y[idx]
    ds.append(roc_auc_score(yy, o_r[idx]) - roc_auc_score(yy, o_b[idx]))
ds = np.array(ds)
print(f"  point delta = {nauc(rmax(RECIPE,'oof'))-base:+.6f}")
print(f"  bootstrap mean={ds.mean():+.6f} sd={ds.std():.6f} "
      f"CI95=[{np.percentile(ds,2.5):+.6f}, {np.percentile(ds,97.5):+.6f}]")
print(f"  P(delta<=0) = {float((ds<=0).mean()):.4f}")

print("\n=== 4. AGGREGATION RULE: max vs rank-mean ===")
for ms, tag in [(MAX3, "max3"), (RECIPE, "recipe")]:
    st_o = np.vstack([arms[m]["oof"] for m in ms])
    mx, mn = st_o.max(0), rankdata(st_o.mean(0)) / st_o.shape[1]
    print(f"  {tag:7s} nested_max={nauc(mx):.6f}  nested_rmean={nauc(mn):.6f}  diff={nauc(mx)-nauc(mn):+.6f}")

print("\n=== 5. SEED-COUNT / bagging honesty of noxb10 ===")
d = np.load(AP / "noxb10.npz", allow_pickle=True)
print(f"  stored seeds={list(np.asarray(d['seeds']))} pool={d['pool']}")
parts = sorted(AP.glob("part_noxb10_s*.npz"))
print(f"  part files on disk = {len(parts)}: {[p.stem.split('_')[-1] for p in parts]}")
po, pt = [], []
for p in parts:
    q = np.load(p)
    po.append(rankdata(q["oof"]) / len(q["oof"]))
    pt.append(rankdata(q["test"]) / len(q["test"]))
reb_o, reb_t = np.mean(po, 0), np.mean(pt, 0)
print(f"  rebuild-from-parts spearman oof={spearmanr(reb_o, d['oof']).correlation:.8f} "
      f"test={spearmanr(reb_t, d['test_pred']).correlation:.8f}")
print(f"  rebuild oof auc={roc_auc_score(y, reb_o):.6f} stored={roc_auc_score(y, d['oof']):.6f}")
