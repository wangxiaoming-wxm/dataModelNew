#!/usr/bin/env python3
"""第二轮换轴：修好的排序损失 + 折内 KNN 密度 + 单查询 YetiRank。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor, Pool
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import build_main, fit_edges_main  # noqa: E402

OUT = ROOT / "vz20" / "artifacts" / "probe_round2"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 3
ITERS = 400
SEED = 2026
LR = 0.03


def auc(y, s):
    return float(roc_auc_score(y, s))


def blend_scan(y, base, arm):
    br, ar = rankdata(base) / len(base), rankdata(arm) / len(arm)
    best = (-1.0, 0.0)
    for w in np.round(np.linspace(0, 1, 21), 2):
        a = auc(y, (1 - w) * br + w * ar)
        if a > best[0]:
            best = (a, float(w))
    return {
        "best_auc": best[0],
        "best_w_arm": best[1],
        "arm_auc": auc(y, arm),
        "delta": best[0] - auc(y, base),
        "spearman": float(spearmanr(base, arm).statistic),
    }


def cv_ranker(X, y, cats, group_id, *, loss="YetiRank"):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        tr = np.asarray(tr)[np.argsort(group_id[tr], kind="stable")]
        va = np.asarray(va)[np.argsort(group_id[va], kind="stable")]
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
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats, group_id=group_id[tr]), verbose=False)
        pred = m.predict(Pool(X.iloc[va], cat_features=cats, group_id=group_id[va]))
        oof[va] = pred
        print(f"    fold{fold} {loss} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  {loss} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_knn(Xnum, y, k=51):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(Xnum, y)):
        scaler = StandardScaler()
        trn = scaler.fit_transform(Xnum[tr])
        van = scaler.transform(Xnum[va])
        m = KNeighborsClassifier(n_neighbors=k, weights="distance")
        m.fit(trn, y[tr])
        oof[va] = m.predict_proba(van)[:, 1]
        print(f"    fold{fold} knn{k} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  knn{k} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_knn_within_source(df, y, k=31):
    """每个 source 内用 (log_days, cond_r) 做 KNN。"""
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    days = pd.to_numeric(df["days"]).to_numpy()
    cond = pd.to_numeric(df["condition"]).fillna(pd.to_numeric(df["condition"]).median()).to_numpy()
    src = df["source"].astype(str).to_numpy()
    X = np.column_stack([np.log1p(days), cond])
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        for s in np.unique(src):
            tr_s = tr[src[tr] == s]
            va_s = va[src[va] == s]
            if len(va_s) == 0:
                continue
            if len(tr_s) < 5 or y[tr_s].sum() == 0 or y[tr_s].sum() == len(tr_s):
                oof[va_s] = y[tr_s].mean() if len(tr_s) else 0.1
                continue
            kk = min(k, len(tr_s))
            scaler = StandardScaler()
            trn = scaler.fit_transform(X[tr_s])
            van = scaler.transform(X[va_s])
            m = KNeighborsClassifier(n_neighbors=kk, weights="distance")
            m.fit(trn, y[tr_s])
            oof[va_s] = m.predict_proba(van)[:, 1]
        print(f"    fold{fold} knn-src {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  knn-src OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def cv_sa_world(train, y):
    """第三世界：source×age 中位数缩放的 cond_r，CatBoost RMSE。"""
    df = train.drop(columns=["label"]).copy()
    scale = df.groupby(["source", "age_range"])["condition"].median()
    idx = pd.MultiIndex.from_frame(df[["source", "age_range"]])
    cond = pd.to_numeric(df["condition"])
    cond_r = (cond / scale.reindex(idx).to_numpy()).fillna(1.0)
    days = pd.to_numeric(df["days"])
    df = df.copy()
    df["condition"] = cond_r  # 用缩放后 condition 走 build_main 会错；自己建瘦特征
    out = pd.DataFrame(index=df.index)
    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["cond_r"] = cond_r.astype(float)
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    ratio = days / cond_r.clip(lower=1e-9)
    out["ratio"] = ratio
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    out["age_range"] = df["age_range"].astype(float)
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    cats = ["region", "source", "month", "version", "age_cat", "grades_c"]
    out["rs"] = out["region"] + "|" + out["source"]
    out["sa"] = out["source"] + "|" + out["age_cat"]
    cats += ["rs", "sa"]
    X = out
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
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
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats), verbose=False)
        oof[va] = m.predict(X.iloc[va])
        print(f"    fold{fold} sa-world {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  sa-world OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    frozen = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    w62 = 0.62 * np.asarray(frozen["main"], float) + 0.38 * np.asarray(frozen["alt"], float)
    print(f"W62 {auc(y, w62):.8f}", flush=True)

    edges = fit_edges_main(train)
    X, cats = build_main(train.drop(columns=["label"]), edges)
    for c in cats:
        X[c] = X[c].astype(str)

    results, oofs = {"w62": auc(y, w62)}, {"w62": w62}

    print("\n== skip cheap KNN (already: blend w=0) ==", flush=True)
    results["knn"] = {"best_w_arm": 0.0, "arm_auc": 0.59576, "note": "prior run"}
    results["knn_src"] = {"best_w_arm": 0.0, "arm_auc": 0.61463, "note": "prior run"}

    print("\n== KNN on build_main numerics ==", flush=True)
    num_cols = [c for c in X.columns if c not in cats]
    Xnum = X[num_cols].apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.fillna(Xnum.median(numeric_only=True)).to_numpy(dtype=float)
    oofs["knn_main"] = cv_knn(Xnum, y, k=51)
    results["knn_main"] = blend_scan(y, w62, oofs["knn_main"])
    print("  vs W62", results["knn_main"], flush=True)

    print("\n== source×age slim world ==", flush=True)
    oofs["sa"] = cv_sa_world(train, y)
    results["sa"] = blend_scan(y, w62, oofs["sa"])
    print("  vs W62", results["sa"], flush=True)

    gid_src = pd.factorize(train["source"].astype(str))[0].astype(np.int32)
    print("\n== YetiRank source (sorted) ==", flush=True)
    try:
        oofs["yeti_src"] = cv_ranker(X, y, cats, gid_src, loss="YetiRank")
        results["yeti_src"] = blend_scan(y, w62, oofs["yeti_src"])
        print("  vs W62", results["yeti_src"], flush=True)
    except Exception as exc:
        results["yeti_src"] = {"error": str(exc)}
        print("  fail", exc, flush=True)

    print("\n== PairLogit source (sorted) ==", flush=True)
    try:
        oofs["pair_src"] = cv_ranker(X, y, cats, gid_src, loss="PairLogit")
        results["pair_src"] = blend_scan(y, w62, oofs["pair_src"])
        print("  vs W62", results["pair_src"], flush=True)
    except Exception as exc:
        results["pair_src"] = {"error": str(exc)}
        print("  fail", exc, flush=True)

    print("\n== YetiRank single query ==", flush=True)
    gid_one = np.zeros(len(y), dtype=np.int32)
    try:
        oofs["yeti_one"] = cv_ranker(X, y, cats, gid_one, loss="YetiRank")
        results["yeti_one"] = blend_scan(y, w62, oofs["yeti_one"])
        print("  vs W62", results["yeti_one"], flush=True)
    except Exception as exc:
        results["yeti_one"] = {"error": str(exc)}
        print("  fail", exc, flush=True)

    print("\n== Quantile 0.9 RMSE-world ==", flush=True)
    try:
        oof_q = np.zeros(len(y), dtype=float)
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
        t0 = time.time()
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            m = CatBoostRegressor(
                loss_function="Quantile:alpha=0.9",
                iterations=ITERS,
                learning_rate=LR,
                depth=5,
                l2_leaf_reg=10,
                verbose=0,
                allow_writing_files=False,
                thread_count=-1,
                random_seed=SEED + fold,
            )
            m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats), verbose=False)
            oof_q[va] = m.predict(X.iloc[va])
            print(f"    fold{fold} q90 {auc(y[va], oof_q[va]):.5f}", flush=True)
        print(f"  q90 OOF={auc(y, oof_q):.5f} ({time.time()-t0:.0f}s)", flush=True)
        oofs["q90"] = oof_q
        results["q90"] = blend_scan(y, w62, oof_q)
        print("  vs W62", results["q90"], flush=True)
    except Exception as exc:
        results["q90"] = {"error": str(exc)}
        print("  fail", exc, flush=True)

    print("\n== YetiRank by region ==", flush=True)
    gid_reg = pd.factorize(train["region"].astype(str))[0].astype(np.int32)
    try:
        oofs["yeti_reg"] = cv_ranker(X, y, cats, gid_reg, loss="YetiRank")
        results["yeti_reg"] = blend_scan(y, w62, oofs["yeti_reg"])
        print("  vs W62", results["yeti_reg"], flush=True)
    except Exception as exc:
        results["yeti_reg"] = {"error": str(exc)}
        print("  fail", exc, flush=True)

    for k, v in oofs.items():
        np.save(OUT / f"oof_{k}.npy", v)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
