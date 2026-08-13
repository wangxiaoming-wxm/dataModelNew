#!/usr/bin/env python3
"""第三轮：CatBoost 分裂准则 / Langevin / Ordered fold_len / Quantile。

同预算 Plain RMSE 对照；融冻结 W62，门禁 +0.001。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import build_main, fit_edges_main  # noqa: E402

OUT = ROOT / "vz20" / "artifacts" / "probe_round3"
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


def cv_rmse(X, y, cats, extra, name):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        kw = dict(
            loss_function="RMSE",
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
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats), verbose=False)
        oof[va] = m.predict(X.iloc[va])
        print(f"    fold{fold} {name} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  {name} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
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

    jobs = [
        ("plain_rmse", {}),
        ("score_L2", {"score_function": "L2"}),
        ("score_NewtonL2", {"score_function": "NewtonL2"}),
        ("score_SolarL2", {"score_function": "SolarL2"}),
        ("score_LOOL2", {"score_function": "LOOL2"}),
        ("score_NewtonCosine", {"score_function": "NewtonCosine"}),
        ("langevin", {"langevin": True, "diffusion_temperature": 10000}),
        ("bernoulli", {"bootstrap_type": "Bernoulli", "subsample": 0.7}),
        ("rsm80", {"rsm": 0.8}),
        ("mvs", {"bootstrap_type": "MVS"}),
        ("leaf_newton", {"leaf_estimation_method": "Newton"}),
        ("q90", {"loss_function": "Quantile:alpha=0.9"}),
        ("huber", {"loss_function": "Huber:delta=1.0"}),
        ("ordered_default", {"boosting_type": "Ordered"}),
        ("ordered_flm15", {"boosting_type": "Ordered", "fold_len_multiplier": 1.5}),
        ("ordered_flm30", {"boosting_type": "Ordered", "fold_len_multiplier": 3.0}),
    ]

    for name, extra in jobs:
        print(f"\n== {name} {extra} ==", flush=True)
        try:
            oof = cv_rmse(X, y, cats, extra, name)
            oofs[name] = oof
            results[name] = blend_scan(y, w62, oof)
            print("  vs W62", results[name], flush=True)
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print("  FAIL", exc, flush=True)

    for k, v in oofs.items():
        np.save(OUT / f"oof_{k}.npy", v)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
