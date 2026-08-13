#!/usr/bin/env python3
"""Probe the one lever the previous rounds never touched.

Removing the categorical crosses costs -0.027 fold-mean, so the entire signal
of this dataset is carried by categorical interactions. CatBoost builds those
automatically via CTR combinations, and the knob controlling how deep it goes
(`max_ctr_complexity`, default 4) was never tuned -- every previous config
left it at the default while hand-writing 81 crosses by hand instead.

Configs are scored on the SELECT seeds only. Whatever wins here still has to
survive the untouched CONFIRM seeds before it can be used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/src")
import features as oldf  # noqa: E402
from vz21_arms import NSPLIT, SELECT_SEEDS  # noqa: E402

ART = Path("/workspace/vz20/artifacts/vz20")
CACHE = ART / "ctr"

CONFIGS = {
    #                       depth l2  rsm  ctr  iters  lr
    "base_d5_l10_ctr4":     (5, 10, 1.0, 4, 800, 0.03),
    "deep_d6_l20_ctr4":     (6, 20, 0.7, 4, 800, 0.03),
    "ctr6_d6_l20":          (6, 20, 0.7, 6, 800, 0.03),
    "ctr2_d6_l20":          (6, 20, 0.7, 2, 800, 0.03),
    "ctr1_d6_l20":          (6, 20, 0.7, 1, 800, 0.03),
    "ctr6_d5_l10":          (5, 10, 1.0, 6, 800, 0.03),
    "ctr6_d6_l20_long":     (6, 20, 0.7, 6, 1600, 0.015),
    "ctr8_d6_l20":          (6, 20, 0.7, 8, 800, 0.03),
    "ctr6_d7_l30":          (7, 30, 0.7, 6, 800, 0.03),
}


def run(name, cfg, Xtr, Xte, cats, y, seed):
    from catboost import CatBoostRegressor, Pool

    CACHE.mkdir(parents=True, exist_ok=True)
    fo, ft = CACHE / f"{name}_s{seed}_oof.npy", CACHE / f"{name}_s{seed}_test.npy"
    if fo.is_file() and ft.is_file():
        return np.load(fo), np.load(ft)
    depth, l2, rsm, ctr, iters, lr = cfg
    folds = list(StratifiedKFold(NSPLIT, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    oof = np.zeros(len(y))
    tep = np.zeros(len(Xte))
    t0 = time.time()
    for tri, vai in folds:
        m = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE", iterations=iters, learning_rate=lr,
            depth=depth, l2_leaf_reg=l2, rsm=rsm, max_ctr_complexity=ctr, random_strength=0.7,
            verbose=0, allow_writing_files=False, one_hot_max_size=2,
            boosting_type="Ordered", random_seed=seed,
        )
        m.fit(Pool(Xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
        oof[vai] = m.predict(Xtr.iloc[vai])
        tep += m.predict(Xte) / len(folds)
    np.save(fo, oof)
    np.save(ft, tep)
    fm = float(np.mean([roc_auc_score(y[v], oof[v]) for _, v in folds]))
    print(f"  {name:<20} seed{seed} foldmean={fm:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return oof, tep


def main():
    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    e = oldf.fit_edges_main(raw)
    X, cats = oldf.build_main(raw, e)
    for c in cats:
        X[c] = X[c].astype(str)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train):].reset_index(drop=True)

    seeds = [int(s) for s in sys.argv[1:]] or list(SELECT_SEEDS)
    res = {}
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        for name, cfg in CONFIGS.items():
            oof, _ = run(name, cfg, Xtr, Xte, cats, y, seed)
            folds = StratifiedKFold(NSPLIT, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
            res.setdefault(name, {})[seed] = float(np.mean([roc_auc_score(y[v], oof[v]) for _, v in folds]))

    print("\n=== summary ===")
    summ = {}
    for name, d in sorted(res.items(), key=lambda t: -np.mean(list(t[1].values()))):
        summ[name] = {"fold_mean": float(np.mean(list(d.values()))), "per_seed": {str(k): v for k, v in d.items()},
                      "config": dict(zip(("depth", "l2", "rsm", "max_ctr_complexity", "iters", "lr"), CONFIGS[name]))}
        print(f"  {name:<20} {summ[name]['fold_mean']:.5f}  {[round(v,5) for v in d.values()]}")
    base = summ["base_d5_l10_ctr4"]["fold_mean"]
    for k in summ:
        summ[k]["delta_vs_vz19_config"] = summ[k]["fold_mean"] - base
    (ART / "vz21_ctr_probe.json").write_text(json.dumps(summ, indent=2) + "\n")
    print("\nwrote", ART / "vz21_ctr_probe.json")


if __name__ == "__main__":
    main()
