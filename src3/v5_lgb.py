"""Strengthen the LightGBM arm under the V2-identical 5-fold protocol.

lgb_te is the most decorrelated arm (rank corr ~0.90 with the CatBoost views)
but sits at 0.671 — too weak for max/mean fusion.  HANDOFF §5.2 says lifting
it above ~0.688 would help more than another CatBoost world.

Changes tried here (all label-honest):
* richer TE column list (the crosses that carry signal in the main world)
* slightly deeper trees / more estimators, still fixed (no early stopping)
* TE smoothing sweep is done by running this script with --smooth

Reports bagged OOF next to the frozen V2 lgb_te score.  Does not touch V2
artefacts and does not read the leaderboard.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import build, fit_edges
from te import encode

TE_RICH = [
    "region", "source", "age_cat", "bin_pat", "month", "version",
    "reg_src", "c10_src", "c5_src", "c20_src",
    "cr10_src", "cr5_src", "cr20_src",
    "d10_reg", "d10_src", "d20_reg", "d20_src",
    "r10_reg", "r10_src", "r20_reg", "r10_age",
    "cr10_reg", "cr10_age", "d10_c10", "d5_c5", "d20_c20",
    "src_c10_age", "reg_c10_age", "d5_reg_src",
    "reg_age", "src_age", "d10_pat", "reg_pat", "dfx_src",
    "dfx_c10", "dfx_cr10", "d10c10_src", "d10c10_reg",
]


def matrix(Xf, Xo, y_fit, seed, smooth):
    num = [c for c in Xf.columns if pd.api.types.is_numeric_dtype(Xf[c])
           and c not in ("cc", "max_g", "V", "x18")]
    Af = Xf[num].to_numpy(dtype=float)
    Ao = [o[num].to_numpy(dtype=float) for o in Xo]
    fit_cols, other_cols = [], [[] for _ in Xo]
    for c in TE_RICH:
        if c not in Xf.columns:
            continue
        ef, eo = encode(Xf[c], y_fit, [o[c] for o in Xo], smoothing=smooth, seed=seed)
        fit_cols.append(ef)
        for i, e in enumerate(eo):
            other_cols[i].append(e)
    Af = np.hstack([Af, np.array(fit_cols).T])
    Ao = [np.hstack([a, np.array(cols).T]) for a, cols in zip(Ao, other_cols)]
    return Af, Ao


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[23000, 23001, 23002, 23003])
    ap.add_argument("--smooth", type=float, default=20.0)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--out", type=Path, default=Path("artifacts/v5_lgb"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw)
    # clean cross2 WITHOUT the noise view — LGB has no ordered TS protection
    X, _ = build(raw, edges, "cross2")
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X[num] = X[num].astype(float).fillna(-999.0)
    Xtr, Xte = X.iloc[: len(train)].reset_index(drop=True), X.iloc[len(train):].reset_index(drop=True)

    oof_seeds, test_parts, per = [], [], []
    t0 = time.time()
    for seed in args.seeds:
        oof = np.zeros(len(y))
        for f, (ti, vi) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y)
        ):
            Af, (Av, At) = matrix(Xtr.iloc[ti], [Xtr.iloc[vi], Xte], y[ti], seed + f, args.smooth)
            m = lgb.LGBMClassifier(
                objective="binary", n_estimators=args.trees, learning_rate=0.02,
                num_leaves=args.leaves, min_child_samples=40, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                verbose=-1, n_jobs=2, random_state=seed + f,
            )
            m.fit(Af, y[ti])
            oof[vi] = m.predict_proba(Av)[:, 1]
            test_parts.append(rankdata(m.predict_proba(At)[:, 1]) / len(At))
        a = float(roc_auc_score(y, oof))
        per.append(a)
        oof_seeds.append(rankdata(oof) / len(oof))
        print(f"  lgb_rich seed={seed} oof={a:.5f} ({time.time()-t0:.0f}s)", flush=True)

    bag = np.mean(oof_seeds, axis=0)
    pred = np.mean(test_parts, axis=0)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "arm_lgb_rich.npz", oof=bag, test=pred, y=y)
    v2 = float(roc_auc_score(y, np.load("artifacts/v2/arm_lgb_te.npz")["oof"]))
    report = {
        "per_seed": per, "bagged_oof_auc": float(roc_auc_score(y, bag)),
        "v2_lgb_te": v2, "delta_vs_v2_lgb": float(roc_auc_score(y, bag) - v2),
        "smooth": args.smooth, "leaves": args.leaves, "trees": args.trees,
        "seeds": args.seeds,
    }
    (args.out / "lgb_rich.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
