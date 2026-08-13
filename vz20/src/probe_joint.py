#!/usr/bin/env python3
"""把 cond_r 世界与 rate 世界拼成单一 CatBoost，看是否优于 rank 融合。"""
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
from train_super714 import (  # noqa: E402
    build_alt,
    build_main,
    fit_edges_alt,
    fit_edges_main,
)

OUT = ROOT / "vz20" / "artifacts" / "probe_joint"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
ITERS = 500
SEED = 2026


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


def cv(X, y, cats, name, ordered=True, depth=5, l2=10):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        kw = dict(
            loss_function="RMSE",
            iterations=ITERS,
            learning_rate=0.03,
            depth=depth,
            l2_leaf_reg=l2,
            random_strength=0.7,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED + fold,
        )
        if ordered:
            kw["boosting_type"] = "Ordered"
        m = CatBoostRegressor(**kw)
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats), verbose=False)
        oof[va] = m.predict(X.iloc[va])
        print(f"    fold{fold} {name} {auc(y[va], oof[va]):.5f}", flush=True)
    print(f"  {name} OOF={auc(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def prefix_frame(df, cats, prefix):
    out = df.copy()
    out.columns = [prefix + str(c) for c in out.columns]
    cats2 = [prefix + c for c in cats]
    return out, cats2


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    frozen = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    w62 = 0.62 * np.asarray(frozen["main"], float) + 0.38 * np.asarray(frozen["alt"], float)
    print(f"W62 {auc(y, w62):.8f}", flush=True)

    raw = train.drop(columns=["label"])
    Xm, cm = build_main(raw, fit_edges_main(train))
    Xa, ca = build_alt(raw, fit_edges_alt(train))
    for c in cm:
        Xm[c] = Xm[c].astype(str)
    for c in ca:
        Xa[c] = Xa[c].astype(str)

    Xap, cap = prefix_frame(Xa, ca, "A_")
    Xj = pd.concat([Xm.reset_index(drop=True), Xap.reset_index(drop=True)], axis=1)
    cj = cm + cap
    for c in cj:
        Xj[c] = Xj[c].astype(str)
    print(f"joint shape {Xj.shape} cats {len(cj)}", flush=True)

    results, oofs = {"w62": auc(y, w62)}, {}

    print("\n== main-only Ordered d5 ==", flush=True)
    oofs["main"] = cv(Xm, y, cm, "main", ordered=True, depth=5, l2=10)
    results["main"] = blend_scan(y, w62, oofs["main"])
    print("  ", results["main"], flush=True)

    print("\n== alt-only Plain d6 ==", flush=True)
    oofs["alt"] = cv(Xa, y, ca, "alt", ordered=False, depth=6, l2=6)
    results["alt"] = blend_scan(y, w62, oofs["alt"])
    print("  ", results["alt"], flush=True)

    mix = 0.62 * rankdata(oofs["main"]) / len(y) + 0.38 * rankdata(oofs["alt"]) / len(y)
    results["mix_same_budget"] = {"arm_auc": auc(y, mix), "delta_vs_w62": auc(y, mix) - auc(y, w62)}
    print("  same-budget W62-style mix", results["mix_same_budget"], flush=True)

    print("\n== joint Ordered d5 ==", flush=True)
    oofs["joint"] = cv(Xj, y, cj, "joint", ordered=True, depth=5, l2=10)
    results["joint"] = blend_scan(y, w62, oofs["joint"])
    results["joint"]["vs_main"] = results["joint"]["arm_auc"] - results["main"]["arm_auc"]
    results["joint"]["vs_mix"] = results["joint"]["arm_auc"] - results["mix_same_budget"]["arm_auc"]
    print("  ", results["joint"], flush=True)

    print("\n== joint Plain d6 ==", flush=True)
    oofs["joint_plain"] = cv(Xj, y, cj, "joint_plain", ordered=False, depth=6, l2=6)
    results["joint_plain"] = blend_scan(y, w62, oofs["joint_plain"])
    results["joint_plain"]["vs_mix"] = results["joint_plain"]["arm_auc"] - results["mix_same_budget"]["arm_auc"]
    print("  ", results["joint_plain"], flush=True)

    for k, v in oofs.items():
        np.save(OUT / f"oof_{k}.npy", v)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
