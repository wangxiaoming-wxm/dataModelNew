"""How high can AUC go on this dataset at all?

Four independent lines of evidence, none of which touches the test labels:

  A. What the observed risk gradient implies.  If a model is calibrated, the
     AUC that the *true* conditional probability function would itself achieve
     is a closed-form functional of the distribution of p(x).  Run it forwards
     (what does our p imply?) and backwards (what would p have to look like for
     AUC = 0.99999?).
  B. Direct decile check: an AUC of ~1 requires the top decile of the score to
     contain essentially every positive.  Measure what it actually contains.
  C. Learning curve: is the 0.70 plateau a data-size problem or a signal
     problem?
  D. Neighbourhood concordance: do rows that are nearly identical in every
     informative feature actually share a label?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260809)


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(pct=True).to_numpy()


def bayes_auc_from_p(p: np.ndarray) -> float:
    """AUC that the true conditional probability p(x) achieves against labels
    drawn as y ~ Bernoulli(p(x)).  Exact, no simulation:

        AUC = [ sum_{i<j, p_i>p_j} p_i(1-p_j) + ... ] / (sum p * sum (1-p))

    Computed in O(n log n) with a sorted prefix sum.
    """
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    q = 1.0 - ps
    # for each i, weight of strictly-smaller p times (1-p)
    csum_q = np.concatenate([[0.0], np.cumsum(q)])
    # handle ties as 0.5 credit
    conc = 0.0
    tie = 0.0
    i = 0
    n = len(ps)
    while i < n:
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        block = slice(i, j + 1)
        below = csum_q[i]                       # sum of (1-p) strictly below
        conc += ps[block].sum() * below
        # within-tie block: all pairs get 0.5
        sp, sq = ps[block].sum(), q[block].sum()
        tie += 0.5 * (sp * sq - (ps[block] * q[block]).sum())
        i = j + 1
    denom = p.sum() * (1.0 - p).sum()
    return float((conc + tie) / denom)


def main() -> None:
    tr = pd.read_csv("data/train.csv")
    y = tr["label"].to_numpy()
    z = np.load("artifacts/v2/arm_cat_d5.npz")
    print("arm keys:", list(z.keys()))

    # best available honest OOF score = the submitted fusion rule
    oof = {}
    for arm in ("cat_d5", "cat_d6", "cat_alt"):
        a = np.load(f"artifacts/v2/arm_{arm}.npz")
        oof[arm] = a["oof"]
    score = np.max(np.stack([rank01(v) for v in oof.values()]), axis=0)
    print(f"fusion OOF AUC = {roc_auc_score(y, score):.5f}")

    # ---- A. calibrated risk -> implied Bayes AUC -------------------------
    # cross-fitted isotonic so the calibration itself is out-of-fold
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    p = np.zeros(len(y))
    for tr_i, te_i in StratifiedKFold(10, shuffle=True, random_state=7).split(score, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(score[tr_i], y[tr_i])
        p[te_i] = iso.predict(score[te_i])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    print("\n[A] cross-fitted calibrated risk p(x)")
    print(f"    mean {p.mean():.4f} (base rate {y.mean():.4f})")
    for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        print(f"    p{q:3d}% = {np.percentile(p, q):.4f}")
    implied = bayes_auc_from_p(p)
    print(f"    AUC a PERFECT model of this risk function would score = {implied:.5f}")

    # backwards: what does AUC = 0.99999 demand?
    print("\n    what p(x) would AUC=0.99999 require?")
    for frac_hi in (0.1002,):
        for hi in (0.5, 0.8, 0.9, 0.99, 0.999, 0.99999):
            lo = (y.mean() - frac_hi * hi) / (1 - frac_hi)
            if lo < 0:
                continue
            pp = np.concatenate([np.full(int(len(y) * frac_hi), hi),
                                 np.full(len(y) - int(len(y) * frac_hi), lo)])
            print(f"      two-point risk (p_hi={hi:<8} on top {frac_hi:.1%}, "
                  f"p_lo={lo:.5f}) -> Bayes AUC {bayes_auc_from_p(pp):.5f}")

    # ---- B. decile realisation ------------------------------------------
    print("\n[B] what the score's top slices actually contain")
    o = np.argsort(-score)
    for k_pct in (1, 2, 5, 10, 20, 50):
        k = int(len(y) * k_pct / 100)
        got = y[o[:k]].sum()
        print(f"    top {k_pct:2d}% ({k:5d} rows): {int(got):4d} of {int(y.sum())} positives "
              f"({got / y.sum():5.1%}), label rate {y[o[:k]].mean():.3f} "
              f"| AUC~1 would need {min(k, int(y.sum())):4d}")

    json.dump({"fusion_oof_auc": float(roc_auc_score(y, score)),
               "implied_bayes_auc": float(implied),
               "p_max": float(p.max()), "p_min": float(p.min())},
              open("hunt/out_h02.json", "w"), indent=2)


if __name__ == "__main__":
    main()
