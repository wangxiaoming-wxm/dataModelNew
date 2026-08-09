"""What is the sharp high-risk tail made of?

The honest OOF score puts a 51% label rate in its top 1% while scoring only
0.699 overall.  That means the risk surface is very steep somewhere.  If some
region of feature space is close to deterministic, finding it explicitly is
worth more than any amount of ensembling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)


def rank01(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def main() -> None:
    tr = pd.read_csv("data/train.csv")
    y = tr["label"].to_numpy()
    oof = [np.load(f"artifacts/v2/arm_{a}.npz")["oof"] for a in ("cat_d5", "cat_d6", "cat_alt")]
    score = np.max(np.stack([rank01(v) for v in oof]), axis=0)

    scale = tr.groupby("source")["condition"].median()
    cond_r = tr["condition"] / tr["source"].map(scale)
    ratio = tr["days"] / cond_r.clip(lower=1e-9)
    tr = tr.assign(cond_r=cond_r, ratio=ratio, score=score)

    o = np.argsort(-score)
    top = tr.iloc[o[:149]]
    rest = tr.iloc[o[149:]]
    print(f"top1% label rate {top.label.mean():.3f}  rest {rest.label.mean():.3f}")

    print("\n--- numeric profile: top 1% vs rest ---")
    for c in ("days", "condition", "cond_r", "ratio", "age_range"):
        print(f"  {c:10s} top median {top[c].median():12.4f}   rest median {rest[c].median():12.4f}"
              f"   top range [{top[c].min():.3f}, {top[c].max():.3f}]")

    print("\n--- categorical profile ---")
    for c in ("source", "region", "month", "version", "grades", "age_range",
              "t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2", "code"):
        vc = top[c].value_counts(normalize=True).head(4)
        base = tr[c].value_counts(normalize=True)
        parts = [f"{k}:{v:.2f}(base {base[k]:.2f})" for k, v in vc.items()]
        print(f"  {c:10s} " + "  ".join(parts))

    # ---- is `ratio` alone enough to find this tail? ----------------------
    print("\n--- label rate by `ratio` decile and by extreme quantiles ---")
    qs = [0, .05, .1, .25, .5, .75, .9, .95, .99, .995, .999, 1.0]
    cut = pd.qcut(tr.ratio, q=qs, duplicates="drop")
    g = tr.groupby(cut, observed=True)["label"].agg(["mean", "size"])
    print(g)

    print("\n--- label rate by days quantile (extremes) ---")
    cutd = pd.qcut(tr.days, q=qs, duplicates="drop")
    print(tr.groupby(cutd, observed=True)["label"].agg(["mean", "size"]))

    print("\n--- label rate by cond_r quantile (extremes) ---")
    cutc = pd.qcut(tr.cond_r, q=qs, duplicates="drop")
    print(tr.groupby(cutc, observed=True)["label"].agg(["mean", "size"]))

    # ---- 2-D cell scan: any near-deterministic cell? ---------------------
    print("\n--- 2D cells (days x cond_r, 12x12) with >=25 rows, sorted by label rate ---")
    db = pd.qcut(tr.days, 12, labels=False, duplicates="drop")
    cb = pd.qcut(tr.cond_r, 12, labels=False, duplicates="drop")
    cell = tr.groupby([db, cb], observed=True)["label"].agg(["mean", "size"])
    cell = cell[cell["size"] >= 25].sort_values("mean", ascending=False)
    print(cell.head(12))
    print("...")
    print(cell.tail(6))

    # ---- how much of the total AUC comes from the extreme tail? ---------
    print("\n--- AUC restricted to subsets (does the model only work in the tail?) ---")
    for lo, hi in ((0, 50), (50, 90), (90, 99), (0, 99), (0, 100)):
        m = (score >= np.percentile(score, lo)) & (score <= np.percentile(score, hi))
        if len(np.unique(y[m])) < 2:
            continue
        print(f"    score pct [{lo:3d},{hi:3d}]: n={m.sum():5d} pos={int(y[m].sum()):4d} "
              f"AUC={roc_auc_score(y[m], score[m]):.4f}")


if __name__ == "__main__":
    main()
