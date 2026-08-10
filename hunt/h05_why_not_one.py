"""How a local AUC of 0.99999 is produced, and why it is worth nothing.

Three ways to put a number above 0.999 on the screen, each of which takes
seconds, plus the one test that separates all of them from a real model: run
the identical pipeline on *shuffled* labels.  A sound pipeline collapses to
0.5.  A leaking one reports the same spectacular number, because the number
never came from the features in the first place.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb


def signal_matrix(df: pd.DataFrame) -> np.ndarray:
    scale = df.groupby("source")["condition"].median()
    cond_r = (df["condition"] / df["source"].map(scale)).fillna(1.0)
    cols = [df["days"], cond_r, np.log((df["days"] / cond_r.clip(lower=1e-9)).clip(lower=1e-9)),
            df["age_range"].astype(float),
            df["region"].astype("category").cat.codes.astype(float),
            df["source"].astype("category").cat.codes.astype(float)]
    cols += [df[b].astype(float) for b in ("t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2")]
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])


def trick_in_sample(X, y):
    """Fit on every row, then score those same rows."""
    m = lgb.LGBMClassifier(objective="binary", n_estimators=3000, learning_rate=0.3,
                           num_leaves=255, min_child_samples=1, min_split_gain=0.0,
                           reg_alpha=0.0, reg_lambda=0.0, verbose=-1, n_jobs=4,
                           random_state=0)
    m.fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def trick_full_data_target_encoding(df, y):
    """Target-encode a near-unique key on the full training set, then 'cross-validate'.

    This is the classic accident.  The encoding column is built once, over all
    rows, so every row's own label is baked into its own feature.  The fold
    split that follows looks completely proper and the OOF number is a fiction.
    """
    key = df["id"]                       # unique per row, like any fine-grained id
    prior, k = y.mean(), 1.0
    stats = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    enc = ((stats["sum"] + prior * k) / (stats["count"] + k))
    feat = key.map(enc).to_numpy()[:, None]

    oof = np.zeros(len(y))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(feat, y):
        m = lgb.LGBMClassifier(objective="binary", n_estimators=100, verbose=-1,
                               n_jobs=4, random_state=0)
        m.fit(feat[ti], y[ti])
        oof[vi] = m.predict_proba(feat[vi])[:, 1]
    return roc_auc_score(y, oof)


def trick_select_then_score(X, y, n_candidates=4000):
    """Generate noise features, keep the ones that correlate, then 'validate'.

    Selection done on all rows before the split leaks the labels just as surely
    as an encoding does.
    """
    rng = np.random.default_rng(0)
    N = rng.normal(size=(len(y), n_candidates))
    corr = np.abs(N.T @ (y - y.mean()))
    keep = np.argsort(-corr)[:300]
    Z = np.hstack([X, N[:, keep]])
    oof = np.zeros(len(y))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        m = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05,
                               num_leaves=63, min_child_samples=5, verbose=-1,
                               n_jobs=4, random_state=0)
        m.fit(Z[ti], y[ti])
        oof[vi] = m.predict_proba(Z[vi])[:, 1]
    return roc_auc_score(y, oof)


def main() -> None:
    df = pd.read_csv("data/train.csv")
    y = df["label"].to_numpy()
    X = signal_matrix(df)
    y_shuf = np.random.default_rng(1).permutation(y)

    rows = []
    for name, fn in (
        ("A. fit on all rows, score the same rows", lambda yy: trick_in_sample(X, yy)),
        ("B. full-data target encoding of `id`, then 5-fold",
         lambda yy: trick_full_data_target_encoding(df, yy)),
        ("C. pick 300 of 4000 noise columns by correlation, then 5-fold",
         lambda yy: trick_select_then_score(X, yy)),
    ):
        real = float(fn(y))
        shuf = float(fn(y_shuf))
        rows.append({"method": name, "reported_auc": real, "auc_on_shuffled_labels": shuf})
        print(f"{name}\n    reported local AUC        = {real:.5f}"
              f"\n    same code, labels shuffled = {shuf:.5f}", flush=True)

    honest = [np.load(f"artifacts/v2/arm_{a}.npz")["oof"] for a in ("cat_d5", "cat_d6", "cat_alt")]
    score = np.max(np.stack([pd.Series(v).rank(pct=True).to_numpy() for v in honest]), axis=0)
    real = float(roc_auc_score(y, score))
    rows.append({"method": "D. this branch's honest pipeline", "reported_auc": real,
                 "auc_on_shuffled_labels": 0.4983})
    print(f"D. this branch's honest pipeline\n    reported local AUC        = {real:.5f}"
          f"\n    same code, labels shuffled = 0.4983  (measured in artifacts/v2/verify.json)")

    print("\nThe shuffled-label column is the whole story: A-C keep their score when the"
          "\nlabels are destroyed, which proves the score never came from the features.")
    json.dump(rows, open("hunt/out_h05.json", "w"), indent=2)


if __name__ == "__main__":
    main()
