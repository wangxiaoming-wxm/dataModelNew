#!/usr/bin/env python3
"""针对 w1=0 弱切片：显式 w1×days×source 交叉，以及按 w1 门控上采样。"""
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

OUT = ROOT / "vz20" / "artifacts" / "probe_w1"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
ITERS = 500
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


def cv(X, y, cats, extra, name, weight=None):
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
            boosting_type="Ordered",
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        kw.update(extra)
        m = CatBoostRegressor(**kw)
        w = None if weight is None else weight[tr]
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats, weight=w), verbose=False)
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
    w1 = train["w1"].to_numpy()
    mask0 = w1 == 0
    print(f"w1=0 slice W62 {auc(y[mask0], w62[mask0]):.5f} n={mask0.sum()}", flush=True)

    edges = fit_edges_main(train)
    X, cats = build_main(train.drop(columns=["label"]), edges)
    for c in cats:
        X[c] = X[c].astype(str)

    results, oofs = {"w62": auc(y, w62)}, {}

    print("\n== control Ordered ==", flush=True)
    oofs["ctrl"] = cv(X, y, cats, {}, "ctrl")
    results["ctrl"] = blend_scan(y, w62, oofs["ctrl"])
    results["ctrl"]["slice_w1_0"] = auc(y[mask0], oofs["ctrl"][mask0])
    print("  ", results["ctrl"], flush=True)

    X2 = X.copy()
    cats2 = list(cats)
    X2["w1c"] = train["w1"].astype(str)
    X2["w2c"] = train["w2"].astype(str)
    X2["w1d5"] = X2["w1c"] + "|" + X2["d5"]
    X2["w1d5s"] = X2["w1d5"] + "|" + X2["source"]
    X2["w2d5s"] = X2["w2c"] + "|" + X2["d5"] + "|" + X2["source"]
    X2["w1w2d5"] = X2["w1c"] + "|" + X2["w2c"] + "|" + X2["d5"]
    cats2 += ["w1c", "w2c", "w1d5", "w1d5s", "w2d5s", "w1w2d5"]
    for c in cats2:
        X2[c] = X2[c].astype(str)

    print("\n== + w1×d5×source crosses ==", flush=True)
    oofs["w1x"] = cv(X2, y, cats2, {}, "w1x")
    results["w1x"] = blend_scan(y, w62, oofs["w1x"])
    results["w1x"]["slice_w1_0"] = auc(y[mask0], oofs["w1x"][mask0])
    results["w1x"]["vs_ctrl"] = results["w1x"]["arm_auc"] - results["ctrl"]["arm_auc"]
    print("  ", results["w1x"], flush=True)

    print("\n== upweight w1=0 x2 ==", flush=True)
    sw = np.where(mask0, 2.0, 1.0)
    oofs["w1up"] = cv(X, y, cats, {}, "w1up", weight=sw)
    results["w1up"] = blend_scan(y, w62, oofs["w1up"])
    results["w1up"]["slice_w1_0"] = auc(y[mask0], oofs["w1up"][mask0])
    results["w1up"]["vs_ctrl"] = results["w1up"]["arm_auc"] - results["ctrl"]["arm_auc"]
    print("  ", results["w1up"], flush=True)

    print("\n== gated splice: w1=0 用上采样臂，其余用 ctrl ==", flush=True)
    gated = oofs["ctrl"].copy()
    # 分位映射到 ctrl 在该切片的分布
    b, s = oofs["ctrl"][mask0], oofs["w1up"][mask0]
    sr = rankdata(s) / (len(s) + 1)
    gated[mask0] = np.quantile(b, sr)
    results["gated"] = {
        "arm_auc": auc(y, gated),
        "delta_vs_ctrl": auc(y, gated) - results["ctrl"]["arm_auc"],
        "delta_vs_w62": auc(y, gated) - auc(y, w62),
        "slice_w1_0": auc(y[mask0], gated[mask0]),
        "slice_ctrl": results["ctrl"]["slice_w1_0"],
        "slice_up": results["w1up"]["slice_w1_0"],
    }
    print("  ", results["gated"], flush=True)

    for k, v in oofs.items():
        np.save(OUT / f"oof_{k}.npy", v)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
