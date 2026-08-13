#!/usr/bin/env python3
"""vz21 arm builder: compute and cache OOF + test predictions for every arm.

Protocol
--------
Outer folds are StratifiedKFold(5) under a set of *outer seeds*. For every
outer seed and every arm we train on the fold-train part and predict the
fold-valid part and the whole test set. Nothing about an arm's configuration
is ever chosen by looking at the rows it is scored on.

Seed roles are fixed in advance and never swapped:

  SELECT_SEEDS  = (424242, 515151)   -- arm screening + blend construction
  CONFIRM_SEEDS = (737373, 848484)   -- touched once, for the final gate only

Caching is per (arm, seed) so the run is resumable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/src")
import features as oldf  # noqa: E402
import vz21_pipeline as newf  # noqa: E402
from vz21_models import te_encode  # noqa: E402

CACHE = Path("/workspace/vz20/artifacts/vz20/arms")
SELECT_SEEDS = (424242, 515151)
CONFIRM_SEEDS = (737373, 848484)
NSPLIT = 5


# ------------------------------------------------------------------ features
def feature_worlds(train, test):
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    ntr = len(train)
    W = {}

    e = oldf.fit_edges_main(raw)
    X, c = oldf.build_main(raw, e)
    for cc in c:
        X[cc] = X[cc].astype(str)
    W["main"] = (X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True), list(c))

    e = oldf.fit_edges_alt(raw)
    X, c = oldf.build_alt(raw, e)
    for cc in c:
        X[cc] = X[cc].astype(str)
    W["alt"] = (X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True), list(c))

    a, b, c = newf.make_matrices(train, test)
    W["new"] = (a, b, list(c))
    return W


# -------------------------------------------------------------------- models
def _cb(Xtr, ytr, Xva, Xte, cats, seed, ordered, depth, l2, rsm, loss="RMSE", iters=800, lr=0.03):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    kw = dict(iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=l2, random_strength=0.7,
              verbose=0, allow_writing_files=False, one_hot_max_size=2, random_seed=seed, rsm=rsm)
    if ordered:
        kw["boosting_type"] = "Ordered"
    if loss == "RMSE":
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **kw)
        m.fit(Pool(Xtr, ytr, cat_features=cats), verbose=False)
        return m.predict(Xva), m.predict(Xte)
    m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **kw)
    m.fit(Pool(Xtr, ytr, cat_features=cats), verbose=False)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def _et(Xtr, ytr, Xva, Xte, cats, seed):
    from sklearn.ensemble import ExtraTreesClassifier

    a, b, c = te_encode(Xtr, ytr, Xva, Xte, cats, seed)
    med = a.median()
    a, b, c = a.fillna(med), b.fillna(med), c.fillna(med)
    m = ExtraTreesClassifier(n_estimators=700, max_features=0.3, min_samples_leaf=15,
                             n_jobs=-1, random_state=seed).fit(a, ytr)
    return m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]


def _lgbte(Xtr, ytr, Xva, Xte, cats, seed):
    import lightgbm as lgb

    a, b, c = te_encode(Xtr, ytr, Xva, Xte, cats, seed)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=800, learning_rate=0.025, num_leaves=15,
                           min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           reg_lambda=10.0, verbose=-1, n_jobs=-1, random_state=seed).fit(a, ytr)
    return m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]


def _glm(Xtr, ytr, Xva, Xte, cats, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    num = [c for c in Xtr.columns if c not in cats]
    A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
    for f in (A, B, C):
        for col in num:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    med = A[num].median()
    for f in (A, B, C):
        f[num] = f[num].fillna(med)
    pre = ColumnTransformer([
        ("num", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25), cats),
    ])
    m = make_pipeline(pre, LogisticRegression(C=0.05, max_iter=3000)).fit(A, ytr)
    return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]


# arm registry: name -> (feature world, callable)
ARMS = {
    # vz19's own two arms, reproduced as the paired baseline
    "A_main_ord":   ("main", lambda *a: _cb(*a, ordered=True,  depth=5, l2=10, rsm=1.0)),
    "A_alt_plain":  ("alt",  lambda *a: _cb(*a, ordered=False, depth=6, l2=6,  rsm=0.3)),
    # same feature worlds, different inductive bias
    "B_main_plain": ("main", lambda *a: _cb(*a, ordered=False, depth=6, l2=6,  rsm=0.5)),
    "B_alt_ord":    ("alt",  lambda *a: _cb(*a, ordered=True,  depth=5, l2=10, rsm=1.0)),
    "C_main_ll":    ("main", lambda *a: _cb(*a, ordered=False, depth=5, l2=8,  rsm=0.5, loss="Logloss")),
    "C_main_deep":  ("main", lambda *a: _cb(*a, ordered=True,  depth=6, l2=20, rsm=0.7)),
    # third feature world (leaner, parsed t3, full x-block)
    "D_new_ord":    ("new",  lambda *a: _cb(*a, ordered=True,  depth=5, l2=10, rsm=1.0)),
    "D_new_plain":  ("new",  lambda *a: _cb(*a, ordered=False, depth=6, l2=6,  rsm=0.3)),
    # non-CatBoost families (genuinely different error structure)
    "E_new_et":     ("new",  lambda *a: _et(*a)),
    "E_main_et":    ("main", lambda *a: _et(*a)),
    "E_main_lgb":   ("main", lambda *a: _lgbte(*a)),
    "F_new_glm":    ("new",  lambda *a: _glm(*a)),
}


def run(arm: str, seed: int, worlds, y, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    fo = CACHE / f"{arm}_s{seed}_oof.npy"
    ft = CACHE / f"{arm}_s{seed}_test.npy"
    if fo.is_file() and ft.is_file() and not force:
        return np.load(fo), np.load(ft)

    world, fn = ARMS[arm]
    Xtr, Xte, cats = worlds[world]
    folds = list(StratifiedKFold(NSPLIT, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    oof = np.zeros(len(y))
    tep = np.zeros(len(Xte))
    t0 = time.time()
    for tri, vai in folds:
        va, te = fn(Xtr.iloc[tri], y[tri], Xtr.iloc[vai], Xte, cats, seed)
        oof[vai] = va
        tep += np.asarray(te) / len(folds)
    np.save(fo, oof)
    np.save(ft, tep)
    fa = [roc_auc_score(y[v], oof[v]) for _, v in folds]
    print(f"  {arm:<14} seed{seed} foldmean={np.mean(fa):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof, tep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SELECT_SEEDS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    worlds = feature_worlds(train, test)
    for k, (a, _b, c) in worlds.items():
        print(f"world {k:<5} feats={a.shape[1]:4d} cats={len(c):3d}")

    for seed in args.seeds:
        print(f"=== outer seed {seed} ===", flush=True)
        for arm in args.arms:
            run(arm, seed, worlds, y, force=args.force)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
