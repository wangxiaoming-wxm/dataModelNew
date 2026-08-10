"""Paired screening harness for CatBoost configurations.

Every configuration is scored on byte-identical folds under the honest protocol
of this branch: fixed iteration count, no early stopping, no look at the outer
validation fold.  One process per configuration with ``thread_count=1`` gets far
better throughput on a 4-core box than one configuration at a time with four
threads, so this is written as a single-config worker.

Usage:
    PYTHONPATH=src2:src3 python3 src3/screen.py --config ordered --seeds 900 901
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from arms import alt2_frame, altboost_frame, catboost_frame
from features import fit_edges, fit_edges_alt, fit_edges_alt2

# Baseline is the `cat_d5` arm that produced the submitted 0.70878 model.
BASE = dict(loss_function="Logloss", learning_rate=0.03, l2_leaf_reg=10,
            random_strength=0.7, depth=5, iterations=1000,
            verbose=False, allow_writing_files=False)

CONFIGS: dict[str, dict] = {
    # --- reference -------------------------------------------------------
    "base":        {},
    "d6":          dict(depth=6, iterations=700, bagging_temperature=1.0),

    # --- ordered boosting: built for exactly this sample size ------------
    "ordered":     dict(boosting_type="Ordered"),
    "ordered_d6":  dict(boosting_type="Ordered", depth=6, iterations=700),
    "ordered_lr02": dict(boosting_type="Ordered", learning_rate=0.02, iterations=1500),

    # --- sampling / regularisation knobs never swept on this branch ------
    "rsm08":       dict(rsm=0.8),
    "rsm06":       dict(rsm=0.6),
    "mvs":         dict(bootstrap_type="MVS", subsample=0.8),
    "bern08":      dict(bootstrap_type="Bernoulli", subsample=0.8),
    "l2_20":       dict(l2_leaf_reg=20),
    "l2_4":        dict(l2_leaf_reg=4),
    "rstr2":       dict(random_strength=2.0),
    "langevin":    dict(langevin=True, diffusion_temperature=10000),
    "newton2":     dict(leaf_estimation_iterations=2),
    "lr02":        dict(learning_rate=0.02, iterations=1500),
    "lr015":       dict(learning_rate=0.015, iterations=2000),
    "depth4":      dict(depth=4, iterations=1400),

    # --- target-statistic configuration ----------------------------------
    "ctr_rich":    dict(simple_ctr=["Borders:CtrBorderCount=15:Prior=0/1:Prior=0.5/1:Prior=1/1",
                                    "Counter:CtrBorderCount=15:Prior=0/1"],
                        combinations_ctr=["Borders:CtrBorderCount=15:Prior=0/1:Prior=0.5/1:Prior=1/1",
                                          "Counter:CtrBorderCount=15:Prior=0/1"]),
    "ctr_prior":   dict(simple_ctr=["Borders:Prior=0/1:Prior=1/1:Prior=2/1:Prior=5/1"],
                        combinations_ctr=["Borders:Prior=0/1:Prior=1/1:Prior=2/1:Prior=5/1"]),
    "ctr_bins50":  dict(simple_ctr=["Borders:CtrBorderCount=50"],
                        combinations_ctr=["Borders:CtrBorderCount=50"]),
}


def build_frame(view: str, raw_all: pd.DataFrame, stream_offset: int):
    edge_fit = {"main": fit_edges, "alt": fit_edges_alt, "alt2": fit_edges_alt2}[view]
    make = {"main": catboost_frame, "alt": altboost_frame, "alt2": alt2_frame}[view]
    return make(raw_all, edge_fit(raw_all), stream_offset=stream_offset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--view", default="main", choices=["main", "alt", "alt2"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("artifacts/screen"))
    args = ap.parse_args()

    if args.config not in CONFIGS:
        raise SystemExit(f"unknown config {args.config}; have {sorted(CONFIGS)}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)

    params = dict(BASE)
    params.update(CONFIGS[args.config])
    params["thread_count"] = args.threads

    t0 = time.time()
    oof_seeds, per_seed = [], []
    for si, seed in enumerate(args.seeds):
        X, cats = build_frame(args.view, raw_all, stream_offset=si + 1)
        Xtr = X.iloc[: len(train)].reset_index(drop=True)
        oof = np.zeros(len(y))
        folds = StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(Xtr, y)
        for f, (ti, vi) in enumerate(folds):
            m = CatBoostClassifier(**params, random_seed=seed + f)
            m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
            oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        a = float(roc_auc_score(y, oof))
        per_seed.append(a)
        oof_seeds.append(rankdata(oof) / len(oof))
        print(f"  {args.config} seed={seed} oof={a:.5f} ({time.time() - t0:.0f}s)", flush=True)

    bag = np.mean(oof_seeds, axis=0)
    res = {"config": args.config, "view": args.view, "folds": args.folds,
           "seeds": args.seeds, "params": {k: str(v) for k, v in CONFIGS[args.config].items()},
           "single_mean": float(np.mean(per_seed)), "per_seed": per_seed,
           "bagged_oof_auc": float(roc_auc_score(y, bag)),
           "elapsed_sec": round(time.time() - t0, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.view}_{args.config}_f{args.folds}"
    np.savez_compressed(args.out / f"oof_{tag}.npz", oof=bag, y=y)
    (args.out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
