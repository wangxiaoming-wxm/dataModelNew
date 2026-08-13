#!/usr/bin/env python3
"""Paired comparison of feature sets under one fixed CatBoost config.

Same outer folds for every variant, so differences are paired and the
fold-to-fold noise (std ~0.017) largely cancels.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/src")
import features as oldf  # noqa: E402
import vz21_pipeline as newf  # noqa: E402

SEEDS = (424242, 515151)
NSPLIT = 5


def catboost_cv(Xtr, cats, y, Xte, folds, depth=5, l2=10, ordered=True, rsm=1.0, iters=800, mseed=0):
    from catboost import CatBoostRegressor, Pool

    oof = np.zeros(len(y))
    tep = np.zeros(len(Xte))
    for tri, vai in folds:
        kw = dict(
            loss_function="RMSE", eval_metric="RMSE", iterations=iters, learning_rate=0.03,
            depth=depth, l2_leaf_reg=l2, random_strength=0.7, verbose=0,
            allow_writing_files=False, one_hot_max_size=2, random_seed=mseed, rsm=rsm,
        )
        if ordered:
            kw["boosting_type"] = "Ordered"
        m = CatBoostRegressor(**kw)
        m.fit(Pool(Xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
        oof[vai] = m.predict(Xtr.iloc[vai])
        tep += m.predict(Xte) / len(folds)
    return oof, tep


def variants(train, test):
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    ntr = len(train)
    out = {}

    e = oldf.fit_edges_main(raw)
    X, c = oldf.build_main(raw, e)
    for cc in c:
        X[cc] = X[cc].astype(str)
    out["OLD_main"] = (X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True), c)

    e = oldf.fit_edges_alt(raw)
    X, c = oldf.build_alt(raw, e)
    for cc in c:
        X[cc] = X[cc].astype(str)
    out["OLD_alt"] = (X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True), c)

    a, b, c = newf.make_matrices(train, test)
    out["NEW_full"] = (a, b, c)

    drop = [f"x{i}" for i in range(19)]
    out["NEW_nox"] = (a.drop(columns=drop), b.drop(columns=drop), c)

    keep_cats = [k for k in c if k not in ("t3_c", "bin_pat", "rs", "d10_r", "d10_s", "c10_s",
                                            "cr10_r", "d10_c10", "r10_s", "region_age", "source_age")]
    dropc = [k for k in c if k not in keep_cats]
    out["NEW_nocross"] = (a.drop(columns=dropc), b.drop(columns=dropc), keep_cats)

    return out


def main():
    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    V = variants(train, test)
    for k, (a, _b, c) in V.items():
        print(f"{k:<14} feats={a.shape[1]:4d} cats={len(c):3d}")

    res = {}
    for seed in SEEDS:
        folds = list(StratifiedKFold(NSPLIT, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
        for name, (a, b, c) in V.items():
            t0 = time.time()
            oof, _ = catboost_cv(a, c, y, b, folds)
            fa = [roc_auc_score(y[v], oof[v]) for _, v in folds]
            res.setdefault(name, {})[seed] = fa
            print(f"seed{seed} {name:<14} foldmean={np.mean(fa):.5f} pooled={roc_auc_score(y,rankdata(oof)):.5f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== summary (fold-mean over all seeds) ===")
    summ = {}
    for name, d in res.items():
        allf = [x for s in d.values() for x in s]
        summ[name] = {"fold_mean": float(np.mean(allf)), "fold_std": float(np.std(allf, ddof=1)), "n": len(allf),
                      "per_seed": {str(k): float(np.mean(v)) for k, v in d.items()}}
        print(f"{name:<14} {summ[name]['fold_mean']:.5f}  (per-seed {summ[name]['per_seed']})")
    base = summ["OLD_main"]["fold_mean"]
    for name in summ:
        summ[name]["delta_vs_OLD_main"] = summ[name]["fold_mean"] - base
    json.dump(summ, open("/workspace/vz20/artifacts/vz20/vz21_featcmp.json", "w"), indent=2)


if __name__ == "__main__":
    main()
