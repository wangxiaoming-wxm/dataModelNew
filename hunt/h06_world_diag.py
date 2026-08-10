"""Why is world 4 weaker than the existing arms?

Compare the worlds feature-by-feature with an honest out-of-fold target
encoding, so a 3000-level cross cannot flatter itself.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src2")
sys.path.insert(0, "src3")

from arms import altboost_frame, catboost_frame            # noqa: E402
from features import fit_edges, fit_edges_alt              # noqa: E402
from worlds import fit_edges_w4, fit_edges_w5, w4_frame, w5_frame  # noqa: E402


def oof_te_auc(col: pd.Series, y: np.ndarray, smoothing: float = 20.0) -> float:
    """AUC of a column encoded by out-of-fold target mean (never sees own label)."""
    v = col.astype(str).to_numpy()
    enc = np.zeros(len(y))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(v, y):
        s = pd.DataFrame({"k": v[ti], "y": y[ti]}).groupby("k")["y"].agg(["sum", "count"])
        prior = y[ti].mean()
        m = (s["sum"] + prior * smoothing) / (s["count"] + smoothing)
        enc[vi] = pd.Series(v[vi]).map(m).fillna(prior).to_numpy()
    return float(roc_auc_score(y, enc))


def num_auc(col: pd.Series, y: np.ndarray) -> float:
    a = roc_auc_score(y, col.astype(float).fillna(col.astype(float).median()))
    return float(max(a, 1 - a))


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    n = len(train)

    builders = {
        "main": (fit_edges, catboost_frame),
        "alt": (fit_edges_alt, altboost_frame),
        "w4": (fit_edges_w4, w4_frame),
        "w5": (fit_edges_w5, w5_frame),
    }
    for name, (fe, mk) in builders.items():
        X, cats = mk(raw, fe(raw), stream_offset=1)
        Xt = X.iloc[:n].reset_index(drop=True)
        nums = [c for c in X.columns if c not in cats]
        scored = []
        for c in nums:
            scored.append((num_auc(Xt[c], y), c, "num"))
        for c in cats:
            if Xt[c].nunique() > 4000:
                continue
            scored.append((oof_te_auc(Xt[c], y), c, "cat"))
        scored.sort(reverse=True)
        print(f"\n=== {name}: {len(nums)} numeric, {len(cats)} categorical ===")
        print("  strongest 12 columns (honest OOF target encoding for categoricals):")
        for a, c, k in scored[:12]:
            print(f"    {a:.4f}  {k}  {c}")
        strong = sum(1 for a, _, _ in scored if a >= 0.58)
        print(f"  columns at AUC >= 0.58: {strong}   >= 0.60: "
              f"{sum(1 for a, _, _ in scored if a >= 0.60)}")


if __name__ == "__main__":
    main()
