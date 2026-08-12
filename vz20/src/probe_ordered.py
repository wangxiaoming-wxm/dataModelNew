#!/usr/bin/env python3
"""确认 Ordered 内部参数：fold_len_multiplier / fold_permutation_block。

协议：同一 5-fold、800 iter、1 seed、1 bag，只改 Ordered 内部超参。
对照默认 Ordered（与 best_v1 main 同族）。晋级：相对默认 +0.001。
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

OUT = ROOT / "vz20" / "artifacts" / "probe_ordered"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
ITERS = 800
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


def cv_ordered(X, y, cats, extra, name):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    fold_aucs = []
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
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats), verbose=False)
        oof[va] = m.predict(X.iloc[va])
        fa = auc(y[va], oof[va])
        fold_aucs.append(fa)
        print(f"    fold{fold} {name} {fa:.5f}", flush=True)
    oof_auc = auc(y, oof)
    print(
        f"  {name} OOF={oof_auc:.5f} mean_fold={np.mean(fold_aucs):.5f} ({time.time()-t0:.0f}s)",
        flush=True,
    )
    return oof, fold_aucs, oof_auc


def add_new_crosses(X, cats):
    X = X.copy()
    cats = list(cats)
    def cross(n, *p):
        s = X[p[0]].astype(str)
        for x in p[1:]:
            s = s + "|" + X[x].astype(str)
        X[n] = s
        cats.append(n)
    cross("msrc", "month", "source")
    cross("vsrc", "version", "source")
    if "d10" in X.columns:
        cross("md10", "month", "d10")
        cross("vd10", "version", "d10")
    return X, cats


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    frozen = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    w62 = 0.62 * np.asarray(frozen["main"], float) + 0.38 * np.asarray(frozen["alt"], float)
    main_o = np.asarray(frozen["main"], float)
    print(f"W62 {auc(y, w62):.8f} frozen_main {auc(y, main_o):.8f}", flush=True)

    edges = fit_edges_main(train)
    X, cats = build_main(train.drop(columns=["label"]), edges)
    for c in cats:
        X[c] = X[c].astype(str)

    results, oofs = {"w62": auc(y, w62), "frozen_main": auc(y, main_o)}, {}

    jobs = [
        ("flm20_default", {}),
        ("flm12", {"fold_len_multiplier": 1.2}),
        ("flm15", {"fold_len_multiplier": 1.5}),
        ("flm25", {"fold_len_multiplier": 2.5}),
        ("flm30", {"fold_len_multiplier": 3.0}),
        ("flm40", {"fold_len_multiplier": 4.0}),
        ("perm1", {"fold_permutation_block": 1}),
        ("perm8", {"fold_permutation_block": 8}),
        ("perm32", {"fold_permutation_block": 32}),
    ]

    for name, extra in jobs:
        print(f"\n== {name} {extra} ==", flush=True)
        try:
            oof, fold_aucs, oof_auc = cv_ordered(X, y, cats, extra, name)
            oofs[name] = oof
            rec = blend_scan(y, w62, oof)
            rec["vs_frozen_main"] = oof_auc - auc(y, main_o)
            rec["fold_aucs"] = fold_aucs
            rec["vs_default"] = None
            results[name] = rec
            print("  vs W62", rec, flush=True)
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print("  FAIL", exc, flush=True)

    if "flm20_default" in oofs:
        base = auc(y, oofs["flm20_default"])
        for name in results:
            if isinstance(results[name], dict) and "arm_auc" in results[name]:
                results[name]["vs_default"] = results[name]["arm_auc"] - base
        print("\n-- vs default Ordered --", flush=True)
        for name, rec in results.items():
            if isinstance(rec, dict) and "vs_default" in rec and rec["vs_default"] is not None:
                print(f"  {name:16s} {rec['arm_auc']:.5f} Δ={rec['vs_default']:+.5f}", flush=True)

    print("\n== extra crosses + default Ordered ==", flush=True)
    X2, cats2 = add_new_crosses(X, cats)
    for c in cats2:
        X2[c] = X2[c].astype(str)
    oof, fold_aucs, oof_auc = cv_ordered(X2, y, cats2, {}, "new_cross")
    oofs["new_cross"] = oof
    rec = blend_scan(y, w62, oof)
    rec["fold_aucs"] = fold_aucs
    rec["vs_default"] = oof_auc - auc(y, oofs["flm20_default"]) if "flm20_default" in oofs else None
    results["new_cross"] = rec
    print("  vs W62", rec, flush=True)

    print("\n== f09d upweight x2 + default Ordered ==", flush=True)
    mask = (train["region"] == "f09d").to_numpy()
    sw = np.where(mask, 2.0, 1.0)
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
            random_strength=0.7,
            boosting_type="Ordered",
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats, weight=sw[tr]), verbose=False)
        oof[va] = m.predict(X.iloc[va])
        print(f"    fold{fold} w_f09d {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  w_f09d OOF={auc(y, oof):.5f} slice={auc(y[mask], oof[mask]):.5f} ({time.time()-t0:.0f}s)", flush=True)
    oofs["w_f09d"] = oof
    rec = blend_scan(y, w62, oof)
    rec["slice_auc"] = auc(y[mask], oof[mask])
    rec["slice_w62"] = auc(y[mask], w62[mask])
    rec["vs_default"] = auc(y, oof) - auc(y, oofs["flm20_default"]) if "flm20_default" in oofs else None
    results["w_f09d"] = rec
    print("  vs W62", rec, flush=True)

    for k, v in oofs.items():
        np.save(OUT / f"oof_{k}.npy", v)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'fold_aucs'}) for k, v in results.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
