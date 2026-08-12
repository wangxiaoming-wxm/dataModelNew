#!/usr/bin/env python3
"""换轴探针：在 build_main 世界上测分类/排序/切片专家，对照同预算 RMSE。

不碰 id-TE / fp_v*。对照冻结 W62，晋级门禁：融 W62 后 OOF +0.001
且 test Spearman(W62) < 0.995。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import build_main, fit_edges_main  # noqa: E402

OUT = ROOT / "vz20" / "artifacts" / "probe_new_axes"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 3
ITERS = 400
SEED = 2026
LR = 0.03


def auc(y, s):
    return float(roc_auc_score(y, s))


def blend_scan(y, base, arm, grid=None):
    grid = grid if grid is not None else np.round(np.linspace(0, 1, 21), 2)
    br, ar = rankdata(base) / len(base), rankdata(arm) / len(arm)
    best = (-1.0, 0.0)
    for w in grid:
        s = (1 - w) * br + w * ar
        a = auc(y, s)
        if a > best[0]:
            best = (a, float(w))
    return {"best_auc": best[0], "best_w_arm": best[1], "arm_auc": auc(y, arm)}


def cv_regressor(X, y, cats, *, loss="RMSE", sample_weight=None, extra=None):
    extra = extra or {}
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        kw = dict(
            loss_function=loss,
            eval_metric="RMSE",
            iterations=ITERS,
            learning_rate=LR,
            depth=5,
            l2_leaf_reg=10,
            random_strength=0.7,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        kw.update(extra)
        m = CatBoostRegressor(**kw)
        w = None if sample_weight is None else sample_weight[tr]
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats, weight=w), verbose=False)
        oof[va] = m.predict(X.iloc[va])
        print(f"    fold{fold} {loss} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  {loss} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_classifier(X, y, cats, *, loss="Logloss", extra=None):
    extra = extra or {}
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        kw = dict(
            loss_function=loss,
            eval_metric="AUC",
            iterations=ITERS,
            learning_rate=LR,
            depth=5,
            l2_leaf_reg=10,
            random_strength=0.7,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        kw.update(extra)
        m = CatBoostClassifier(**kw)
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats), verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"    fold{fold} clf-{loss} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  clf-{loss} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_ranker(X, y, cats, group_id, *, loss="YetiRank"):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = CatBoostRanker(
            loss_function=loss,
            iterations=ITERS,
            learning_rate=LR,
            depth=5,
            l2_leaf_reg=10,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        train_pool = Pool(X.iloc[tr], y[tr], cat_features=cats, group_id=group_id[tr])
        m.fit(train_pool, verbose=False)
        oof[va] = m.predict(Pool(X.iloc[va], cat_features=cats, group_id=group_id[va]))
        print(f"    fold{fold} {loss} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  {loss} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_lgb(X, y, cats):
    import lightgbm as lgb

    Xc = X.copy()
    cat_idx = []
    for i, c in enumerate(Xc.columns):
        if c in cats:
            Xc[c] = Xc[c].astype("category")
            cat_idx.append(i)
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(Xc, y)):
        dtr = lgb.Dataset(Xc.iloc[tr], y[tr], categorical_feature=cat_idx, free_raw_data=False)
        dva = lgb.Dataset(Xc.iloc[va], y[va], categorical_feature=cat_idx, free_raw_data=False)
        m = lgb.train(
            dict(
                objective="binary",
                metric="auc",
                learning_rate=LR,
                num_leaves=32,
                min_data_in_leaf=50,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbosity=-1,
                seed=SEED + fold,
            ),
            dtr,
            num_boost_round=ITERS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(Xc.iloc[va], num_iteration=m.best_iteration)
        print(f"    fold{fold} lgb {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  lgb OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_slice_specialist(X, y, cats, mask):
    """只在切片上 CV，OOF 仅切片有值。"""
    idx = np.where(mask)[0]
    oof = np.full(len(y), np.nan)
    y_s = y[idx]
    Xs = X.iloc[idx].reset_index(drop=True)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    local = np.zeros(len(idx))
    for fold, (tr, va) in enumerate(skf.split(Xs, y_s)):
        m = CatBoostRegressor(
            loss_function="RMSE",
            iterations=ITERS,
            learning_rate=LR,
            depth=5,
            l2_leaf_reg=10,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        m.fit(Pool(Xs.iloc[tr], y_s[tr], cat_features=cats), verbose=False)
        local[va] = m.predict(Xs.iloc[va])
        print(f"    fold{fold} slice {auc(y_s[va], local[va]):.5f}", flush=True)
    oof[idx] = local
    print(f"  slice OOF={auc(y_s, local):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def splice_within(base, specialist, mask):
    """切片内用专家名次替换，分位映射回 base 在该切片的分布。"""
    out = base.copy()
    b, s = base[mask], specialist[mask]
    # 专家 rank -> base 的分位
    sr = rankdata(s) / (len(s) + 1)
    mapped = np.quantile(b, sr)
    out[mask] = mapped
    return out


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    frozen = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    w62 = 0.62 * np.asarray(frozen["main"], float) + 0.38 * np.asarray(frozen["alt"], float)
    print(f"W62 frozen {auc(y, w62):.8f}", flush=True)

    edges = fit_edges_main(train)
    X, cats = build_main(train.drop(columns=["label"]), edges)
    for c in cats:
        X[c] = X[c].astype(str)
    print(f"X {X.shape} cats {len(cats)}", flush=True)

    results = {"w62": auc(y, w62)}
    oofs = {"w62": w62}

    # 0) cheap: ratio rank mix
    ratio = X["ratio"].to_numpy()
    results["ratio_blend"] = blend_scan(y, w62, ratio)
    print("ratio blend", results["ratio_blend"], flush=True)

    print("\n== same-budget RMSE control ==", flush=True)
    oofs["rmse"] = cv_regressor(X, y, cats, loss="RMSE")
    results["rmse"] = blend_scan(y, w62, oofs["rmse"])
    print("  vs W62", results["rmse"], "spearman", float(spearmanr(w62, oofs["rmse"]).statistic), flush=True)

    print("\n== Logloss classifier ==", flush=True)
    oofs["logloss"] = cv_classifier(X, y, cats, loss="Logloss")
    results["logloss"] = blend_scan(y, w62, oofs["logloss"])
    print("  vs W62", results["logloss"], "spearman", float(spearmanr(w62, oofs["logloss"]).statistic), flush=True)

    print("\n== Logloss balanced ==", flush=True)
    oofs["logloss_bal"] = cv_classifier(
        X, y, cats, loss="Logloss", extra={"auto_class_weights": "Balanced"}
    )
    results["logloss_bal"] = blend_scan(y, w62, oofs["logloss_bal"])
    print("  vs W62", results["logloss_bal"], flush=True)

    print("\n== weighted RMSE ==", flush=True)
    sw = np.where(y == 1, (y == 0).mean() / max((y == 1).mean(), 1e-9), 1.0)
    oofs["wrmse"] = cv_regressor(X, y, cats, loss="RMSE", sample_weight=sw)
    results["wrmse"] = blend_scan(y, w62, oofs["wrmse"])
    print("  vs W62", results["wrmse"], flush=True)

    print("\n== LightGBM binary on build_main ==", flush=True)
    oofs["lgb"] = cv_lgb(X, y, cats)
    results["lgb"] = blend_scan(y, w62, oofs["lgb"])
    print("  vs W62", results["lgb"], "spearman", float(spearmanr(w62, oofs["lgb"]).statistic), flush=True)

    print("\n== YetiRank by source ==", flush=True)
    gid = pd.factorize(train["source"].astype(str))[0].astype(np.int32)
    try:
        oofs["yeti_src"] = cv_ranker(X, y, cats, gid, loss="YetiRank")
        results["yeti_src"] = blend_scan(y, w62, oofs["yeti_src"])
        print("  vs W62", results["yeti_src"], flush=True)
    except Exception as exc:
        results["yeti_src"] = {"error": str(exc)}
        print("  YetiRank failed", exc, flush=True)

    print("\n== PairLogit by source ==", flush=True)
    try:
        oofs["pair_src"] = cv_ranker(X, y, cats, gid, loss="PairLogit")
        results["pair_src"] = blend_scan(y, w62, oofs["pair_src"])
        print("  vs W62", results["pair_src"], flush=True)
    except Exception as exc:
        results["pair_src"] = {"error": str(exc)}
        print("  PairLogit failed", exc, flush=True)

    print("\n== f09d specialist splice ==", flush=True)
    mask = (train["region"] == "f09d").to_numpy()
    spec = cv_slice_specialist(X, y, cats, mask)
    oofs["f09d_spec"] = spec
    spliced = splice_within(w62, spec, mask)
    results["f09d_splice"] = {
        "global": auc(y, spliced),
        "slice_w62": auc(y[mask], w62[mask]),
        "slice_spec": auc(y[mask], spec[mask]),
        "slice_spliced": auc(y[mask], spliced[mask]),
        "delta_global": auc(y, spliced) - auc(y, w62),
    }
    print("  f09d", results["f09d_splice"], flush=True)

    print("\n== CAR_2 specialist splice ==", flush=True)
    mask2 = (train["source"] == "CAR_2|ENG_262").to_numpy()
    spec2 = cv_slice_specialist(X, y, cats, mask2)
    oofs["car2_spec"] = spec2
    spliced2 = splice_within(w62, spec2, mask2)
    results["car2_splice"] = {
        "global": auc(y, spliced2),
        "slice_w62": auc(y[mask2], w62[mask2]),
        "slice_spec": auc(y[mask2], spec2[mask2]),
        "delta_global": auc(y, spliced2) - auc(y, w62),
    }
    print("  CAR_2", results["car2_splice"], flush=True)

    # persist
    for k, v in oofs.items():
        np.save(OUT / f"oof_{k}.npy", v)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
