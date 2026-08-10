"""Independent read-only audit of the V4max3pro submission.

Recomputes: arm OOF AUCs, the 5-block nested ruler, max3 baseline, the claimed
5-arm recipe, spearman vs the frozen max3 submission, and whether
submissions/submission_v4max3pro.csv is byte/rank reproducible.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace")
DATA = ROOT / "data"
SUB = ROOT / "submissions"
A3 = ROOT / "artifacts" / "v4max3"
AP = ROOT / "artifacts" / "v4max3pro"

y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
test_ids = pd.read_csv(DATA / "test.csv")["id"].values
N_TR, N_TE = len(y), len(test_ids)
print(f"train n={N_TR} pos={y.sum()} rate={y.mean():.5f}  test n={N_TE}")


def load(path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], float)
    te = np.asarray(d["test_pred"] if "test_pred" in d else d["test"], float)
    extra = {k: d[k] for k in d.files if k not in ("oof", "test_pred", "test")}
    return oof, te, extra


ARMS = {
    "merger_ord8": A3 / "merger_ord8.npz",
    "v2_cat_alt8": A3 / "v2_cat_alt8.npz",
    "ord_noxb_bag": A3 / "ord_noxb_bag.npz",
    "ordered_bag": A3 / "ordered_bag.npz",
    "plus_v10": A3 / "plus_v10.npz",
    "b7_closest": A3 / "b7_closest.npz",
    "mine_noxb8": A3 / "mine_noxb8.npz",
    "plus_strong": AP / "plus_strong.npz",
    "noxb10": AP / "noxb10.npz",
    "main10_ord_es": AP / "main10_ord_es.npz",
    "hybrid10": AP / "hybrid10.npz",
    "alt10": AP / "alt10.npz",
    "merger_ord_es": AP / "merger_ord_es.npz",
}

arms = {}
print("\n=== ARM INVENTORY (my own AUC recompute) ===")
for name, p in ARMS.items():
    if not p.exists():
        print(f"  MISSING {name}: {p}")
        continue
    oof, te, extra = load(p)
    if len(oof) != N_TR or len(te) != N_TE:
        print(f"  SHAPE-BAD {name}: {oof.shape} {te.shape}")
        continue
    arms[name] = {
        "oof": rankdata(oof) / len(oof),
        "te": rankdata(te) / len(te),
        "oof_raw": oof,
    }
    seeds = extra.get("seeds")
    per = extra.get("per_seed")
    ystored = extra.get("y")
    ymatch = "n/a" if ystored is None else str(bool(np.array_equal(np.asarray(ystored, int), y)))
    print(
        f"  {name:16s} oof_auc={roc_auc_score(y, oof):.6f} "
        f"y_matches_train={ymatch} seeds={None if seeds is None else list(np.asarray(seeds))}"
    )
    if per is not None:
        print(f"      per_seed={['%.5f' % a for a in np.asarray(per, float)]}")


def nested_auc(oof, n_blocks=5):
    out = np.zeros(len(oof))
    for b in np.array_split(np.arange(len(oof)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def rmax(members, which):
    return np.vstack([arms[m][which] for m in members]).max(axis=0)


MAX3 = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
RECIPE = MAX3 + ["plus_strong", "noxb10"]

max3_oof, max3_te = rmax(MAX3, "oof"), rmax(MAX3, "te")
max3_nested, max3_full = nested_auc(max3_oof), float(roc_auc_score(y, max3_oof))
print("\n=== BASELINE max3 ===")
print(f"  nested(5blk)={max3_nested:.6f}  full_oof={max3_full:.6f}")

base_sub = pd.read_csv(SUB / "submission_v4_max3.csv")
print(f"  submission_v4_max3.csv rows={len(base_sub)} cols={list(base_sub.columns)}")
print(f"  id order identical to test.csv: {np.array_equal(base_sub['id'].values, test_ids)}")
sp_base = spearmanr(max3_te, base_sub["label"].values).correlation
print(f"  spearman(recomputed max3 test, submission_v4_max3.csv) = {sp_base:.8f}")

print("\n=== CLAIMED RECIPE max5 ===")
missing = [m for m in RECIPE if m not in arms]
if missing:
    print(f"  CANNOT BUILD, missing {missing}")
else:
    r_oof, r_te = rmax(RECIPE, "oof"), rmax(RECIPE, "te")
    r_nested, r_full = nested_auc(r_oof), float(roc_auc_score(y, r_oof))
    print(f"  nested={r_nested:.6f}  full_oof={r_full:.6f}  delta_vs_max3={r_nested-max3_nested:+.6f}")
    print(f"  spearman(recipe_te, max3_te) = {spearmanr(r_te, max3_te).correlation:.6f}")

    cand = pd.read_csv(SUB / "submission_v4max3pro.csv")
    print(f"\n  submission_v4max3pro.csv rows={len(cand)} cols={list(cand.columns)}")
    print(f"  id order identical to test.csv: {np.array_equal(cand['id'].values, test_ids)}")
    cl = cand["label"].values
    print(f"  label range [{cl.min():.6f}, {cl.max():.6f}]  unique={len(np.unique(cl))}")
    sp_rec = spearmanr(np.clip(r_te, 0.001, 0.999), cl).correlation
    print(f"  >>> spearman(claimed recipe rebuild, submission file) = {sp_rec:.8f}")
    print(f"  max abs diff = {np.abs(np.clip(r_te,0.001,0.999)-cl).max():.3e}")
    print(f"  exact equal (1e-12) = {np.allclose(np.clip(r_te,0.001,0.999), cl, atol=1e-12)}")
    print(f"  spearman(submission file, submission_v4_max3.csv) = "
          f"{spearmanr(cl, base_sub['label'].values).correlation:.6f}")

    bsf = pd.read_csv(SUB / "submission_v4max3pro_best_so_far.csv")
    print(f"  identical to _best_so_far.csv: "
          f"{np.allclose(bsf['label'].values, cl, atol=1e-12)}")

    # shuffled-label sanity
    rng = np.random.default_rng(0)
    print(f"  shuffled-label OOF AUC = {roc_auc_score(rng.permutation(y), r_oof):.6f}")

    # honest-only subset (drop every ES / 10-fold arm)
    honest = ["merger_ord8", "v2_cat_alt8"]
    print(f"\n  honest-only max2 nested = {nested_auc(rmax(honest,'oof')):.6f}")

print("\n=== NESTED RULER SENSITIVITY (max3 vs recipe under other block counts) ===")
for nb in (1, 2, 3, 5, 8, 10, 20):
    a = nested_auc(max3_oof, nb)
    b = nested_auc(rmax(RECIPE, "oof"), nb) if not missing else float("nan")
    print(f"  blocks={nb:3d}  max3={a:.6f}  recipe={b:.6f}  delta={b-a:+.6f}")

print("\n=== SELECTION-BIAS PROBE: how many candidate recipes beat the gate? ===")
names = [n for n in arms]
res = []
for r in range(2, 6):
    for c in combinations(names, r):
        if sum(1 for x in c if x in MAX3) < 2:
            continue
        res.append((nested_auc(rmax(c, "oof")), "+".join(c)))
res.sort(reverse=True)
print(f"  enumerated candidates = {len(res)}")
print(f"  best  = {res[0][0]:.6f}  {res[0][1]}")
gate = max3_nested + 0.0015
print(f"  #candidates with nested >= max3+0.0015 ({gate:.6f}) = {sum(1 for s,_ in res if s>=gate)}")
print(f"  #candidates with nested >= max3+0.0021 = {sum(1 for s,_ in res if s>=max3_nested+0.0021)}")
print("  top 8:")
for s, lab in res[:8]:
    print(f"    {s:.6f}  {lab}")
rank_of_recipe = next((i for i, (_, lab) in enumerate(res) if set(lab.split("+")) == set(RECIPE)), None)
print(f"  rank of claimed recipe among enumerated = {rank_of_recipe}")

print("\n=== CV->LB EXTRAPOLATION CHECK ===")
gap = 0.71222 - max3_nested
print(f"  gap(max3) = 0.71222 - {max3_nested:.6f} = {gap:.6f}")
if not missing:
    print(f"  exp_lb(recipe) = {r_nested:.6f} + {gap:.6f} = {r_nested+gap:.6f}")
    print(f"  needed nested for 0.7155 = {0.7155-gap:.6f}  -> shortfall {0.7155-gap-r_nested:+.6f}")

print("\n=== STATUS REPORT CROSS-CHECK ===")
sr = json.loads((AP / "status_report.json").read_text())
mine = {
    "max3_nested": max3_nested,
    "gap": gap,
    "needed_nested": 0.7155 - gap,
    "best.nested": r_nested if not missing else None,
    "best.delta": (r_nested - max3_nested) if not missing else None,
    "best.exp_lb": (r_nested + gap) if not missing else None,
    "best.sp": spearmanr(r_te, max3_te).correlation if not missing else None,
    "noxb10_oof": float(roc_auc_score(y, arms["noxb10"]["oof_raw"])) if "noxb10" in arms else None,
}
claimed = {
    "max3_nested": sr["max3_nested"],
    "gap": sr["gap"],
    "needed_nested": sr["needed_nested"],
    "best.nested": sr["best"]["nested"],
    "best.delta": sr["best"]["delta"],
    "best.exp_lb": sr["best"]["exp_lb"],
    "best.sp": sr["best"]["sp"],
    "noxb10_oof": sr["noxb10_oof"],
}
for k in mine:
    m, c = mine[k], claimed[k]
    flag = "OK " if (m is not None and abs(m - c) < 5e-6) else "MISMATCH"
    print(f"  {flag} {k:16s} mine={m if m is None else f'{m:.6f}'}  claimed={c:.6f}")
