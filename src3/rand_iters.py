"""Random-iteration diversity without peeking at the validation fold.

docs/HANDOFF.md §5.3 notes that B7's early-stopping may have bought bagging
diversity through varying tree counts rather than through the early-stop
itself.  This experiment isolates that effect: each (seed, fold) draws a tree
count from a fixed range *before* looking at any fold, so the outer validation
fold is never used for selection.
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

from arms import catboost_frame
from features import fit_edges

BASE = dict(loss_function="Logloss", learning_rate=0.03, l2_leaf_reg=10,
            random_strength=0.7, depth=5, verbose=False,
            allow_writing_files=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[910, 911, 912, 913])
    ap.add_argument("--lo", type=int, default=400)
    ap.add_argument("--hi", type=int, default=1200)
    ap.add_argument("--fixed", type=int, default=1000,
                    help="paired control: same folds, fixed iteration count")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("artifacts/screen"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw)

    results = {"fixed": [], "rand": [], "iters": []}
    oof_fixed_seeds, oof_rand_seeds = [], []
    t0 = time.time()

    for si, seed in enumerate(args.seeds):
        X, cats = catboost_frame(raw, edges, stream_offset=si + 1)
        Xtr = X.iloc[: len(train)].reset_index(drop=True)
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(Xtr, y))
        oof_f = np.zeros(len(y))
        oof_r = np.zeros(len(y))
        rng = np.random.default_rng(seed)   # iteration draws are pre-fold
        for f, (ti, vi) in enumerate(folds):
            n_trees = int(rng.integers(args.lo, args.hi + 1))
            results["iters"].append(n_trees)
            for label, n_it, oof in (("fixed", args.fixed, oof_f), ("rand", n_trees, oof_r)):
                m = CatBoostClassifier(**BASE, iterations=n_it, thread_count=args.threads,
                                       random_seed=seed + f)
                m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
                oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        af = float(roc_auc_score(y, oof_f))
        ar = float(roc_auc_score(y, oof_r))
        results["fixed"].append(af)
        results["rand"].append(ar)
        oof_fixed_seeds.append(rankdata(oof_f) / len(oof_f))
        oof_rand_seeds.append(rankdata(oof_r) / len(oof_r))
        print(f"  seed={seed} fixed={af:.5f} rand={ar:.5f} "
              f"delta={ar - af:+.5f} ({time.time() - t0:.0f}s)", flush=True)

    bag_f = float(roc_auc_score(y, np.mean(oof_fixed_seeds, axis=0)))
    bag_r = float(roc_auc_score(y, np.mean(oof_rand_seeds, axis=0)))
    out = {
        "seeds": args.seeds, "lo": args.lo, "hi": args.hi, "fixed": args.fixed,
        "per_seed_fixed": results["fixed"], "per_seed_rand": results["rand"],
        "iters_drawn": results["iters"],
        "single_mean_fixed": float(np.mean(results["fixed"])),
        "single_mean_rand": float(np.mean(results["rand"])),
        "bagged_fixed": bag_f, "bagged_rand": bag_r,
        "delta_bagged": bag_r - bag_f,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rand_iters.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
