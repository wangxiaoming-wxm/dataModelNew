"""Model zoo for vz21.

Every learner exposes the same contract:

    fit_predict(Xtr, ytr, Xva, Xte, cats, levels, seed) -> (va_score, te_score)

Scores are returned raw; the ensemble rank-normalises them. No learner ever
sees the validation labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from vz21_pipeline import as_category, as_ordinal

TE_SMOOTH = 25.0
TE_INNER = 5


def te_encode(Xtr, ytr, Xva, Xte, cats, seed=0, smooth=TE_SMOOTH):
    """Fold-internal target encoding for learners without native categoricals.

    The training rows get an inner-OOF encoding (their own label never enters
    their own code), while validation/test rows are encoded with the table
    fitted on the full training part. This is the same discipline CatBoost
    applies internally, and it is what makes LightGBM/XGBoost competitive here.
    """
    prior = float(np.mean(ytr))
    A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
    y = np.asarray(ytr, dtype=float)
    inner = list(StratifiedKFold(TE_INNER, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    for c in cats:
        ktr = Xtr[c].to_numpy(str)
        uniq, codes = np.unique(ktr, return_inverse=True)
        index = {k: i for i, k in enumerate(uniq)}
        n = len(uniq)

        oof = np.full(len(y), prior)
        for a, b in inner:
            s = np.bincount(codes[a], weights=y[a], minlength=n)
            cnt = np.bincount(codes[a], minlength=n).astype(float)
            p = float(y[a].mean())
            tab = (s + smooth * p) / (cnt + smooth)
            v = tab[codes[b]]
            v[cnt[codes[b]] == 0] = p
            oof[b] = v

        s = np.bincount(codes, weights=y, minlength=n)
        cnt = np.bincount(codes, minlength=n).astype(float)
        table = (s + smooth * prior) / (cnt + smooth)

        A[c] = oof
        A[f"{c}__cnt"] = cnt[codes]
        for frame, src in ((B, Xva), (C, Xte)):
            k = src[c].to_numpy(str)
            idx = np.array([index.get(v, -1) for v in k])
            frame[c] = np.where(idx >= 0, table[np.clip(idx, 0, n - 1)], prior)
            frame[f"{c}__cnt"] = np.where(idx >= 0, cnt[np.clip(idx, 0, n - 1)], 0.0)

    for f in (A, B, C):
        for col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    return A, B, C


# ------------------------------------------------------------------ CatBoost
def _catboost(Xtr, ytr, Xva, Xte, cats, seed, *, ordered, depth, l2, rsm, iters=800, lr=0.03, loss="RMSE"):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    kw = dict(
        iterations=iters,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        random_strength=0.7,
        verbose=0,
        allow_writing_files=False,
        one_hot_max_size=2,
        random_seed=seed,
        rsm=rsm,
    )
    if ordered:
        kw["boosting_type"] = "Ordered"
    if loss == "RMSE":
        model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **kw)
        pred = lambda m, X: m.predict(X)  # noqa: E731
    else:
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **kw)
        pred = lambda m, X: m.predict_proba(X)[:, 1]  # noqa: E731
    model.fit(Pool(Xtr, ytr, cat_features=cats), verbose=False)
    return pred(model, Xva), pred(model, Xte)


def cb_ord(Xtr, ytr, Xva, Xte, cats, levels, seed):
    return _catboost(Xtr, ytr, Xva, Xte, cats, seed, ordered=True, depth=5, l2=10, rsm=1.0)


def cb_plain(Xtr, ytr, Xva, Xte, cats, levels, seed):
    return _catboost(Xtr, ytr, Xva, Xte, cats, seed, ordered=False, depth=6, l2=6, rsm=0.3)


def cb_logloss(Xtr, ytr, Xva, Xte, cats, levels, seed):
    return _catboost(Xtr, ytr, Xva, Xte, cats, seed, ordered=False, depth=5, l2=8, rsm=0.5, loss="Logloss")


# ------------------------------------------------------------------ LightGBM
def _lgb(Xtr, ytr, Xva, Xte, cats, levels, seed, native=False, **params):
    import lightgbm as lgb

    if native:
        a, _ = as_category(Xtr, cats, levels)
        b, _ = as_category(Xva, cats, levels)
        c, _ = as_category(Xte, cats, levels)
    else:
        a, b, c = te_encode(Xtr, ytr, Xva, Xte, cats, seed)
    base = dict(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        max_cat_to_onehot=4,
        cat_smooth=30.0,
        cat_l2=20.0,
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )
    base.update(params)
    m = lgb.LGBMClassifier(**base)
    m.fit(a, ytr, categorical_feature=cats if native else "auto")
    return m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]


def lgb_a(Xtr, ytr, Xva, Xte, cats, levels, seed):
    return _lgb(Xtr, ytr, Xva, Xte, cats, levels, seed)


def lgb_b(Xtr, ytr, Xva, Xte, cats, levels, seed):
    """Deeper, stronger feature subsampling -- decorrelated from lgb_a."""
    return _lgb(
        Xtr, ytr, Xva, Xte, cats, levels, seed,
        num_leaves=31, min_child_samples=100, colsample_bytree=0.35,
        reg_lambda=20.0, n_estimators=900, learning_rate=0.025,
    )


def lgb_native(Xtr, ytr, Xva, Xte, cats, levels, seed):
    return _lgb(Xtr, ytr, Xva, Xte, cats, levels, seed, native=True)


# ------------------------------------------------------------------- XGBoost
def xgb_a(Xtr, ytr, Xva, Xte, cats, levels, seed):
    import xgboost as xgb

    a, b, c = te_encode(Xtr, ytr, Xva, Xte, cats, seed)
    m = xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        tree_method="hist",
        max_bin=128,
        eval_metric="auc",
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(a, ytr, verbose=False)
    return m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]


# --------------------------------------------------- sklearn HistGradientBoost
def hgb_a(Xtr, ytr, Xva, Xte, cats, levels, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    a, b, c = te_encode(Xtr, ytr, Xva, Xte, cats, seed)
    m = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=5.0,
        max_features=0.6,
        random_state=seed,
    )
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]


# --------------------------------------------------------- random forest / ET
def et_a(Xtr, ytr, Xva, Xte, cats, levels, seed):
    """Extremely randomised trees on TE features -- high-variance, low-bias
    learner whose errors decorrelate well from boosted trees."""
    from sklearn.ensemble import ExtraTreesClassifier

    a, b, c = te_encode(Xtr, ytr, Xva, Xte, cats, seed)
    med = a.median()
    a, b, c = a.fillna(med), b.fillna(med), c.fillna(med)
    m = ExtraTreesClassifier(
        n_estimators=600,
        max_features=0.3,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]


# ------------------------------------------------- regularised linear (GLM)
def glm_a(Xtr, ytr, Xva, Xte, cats, levels, seed):
    """One-hot + spline-free logistic ridge. A genuinely different inductive
    bias from every tree above, which is the point of including it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.compose import ColumnTransformer

    num = [c for c in Xtr.columns if c not in cats]
    pre = ColumnTransformer(
        [
            ("num", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25), cats),
        ]
    )
    m = make_pipeline(pre, LogisticRegression(C=0.05, max_iter=3000, solver="lbfgs"))
    A = Xtr.copy()
    B = Xva.copy()
    C = Xte.copy()
    for f in (A, B, C):
        for col in num:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    A[num] = A[num].fillna(A[num].median())
    B[num] = B[num].fillna(A[num].median())
    C[num] = C[num].fillna(A[num].median())
    m.fit(A, ytr)
    return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]


ZOO = {
    "cb_ord": cb_ord,
    "cb_plain": cb_plain,
    "cb_logloss": cb_logloss,
    "lgb_a": lgb_a,
    "lgb_b": lgb_b,
    "lgb_native": lgb_native,
    "xgb_a": xgb_a,
    "hgb_a": hgb_a,
    "et_a": et_a,
    "glm_a": glm_a,
}
