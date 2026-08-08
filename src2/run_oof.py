"""Compute out-of-fold and test predictions for every arm on shared partitions.

Loops seeds on the outside so the (expensive) feature frame is built once per
seed and reused by every arm; all arms therefore see byte-identical folds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from arms import ARMS, catboost_frame, fit_predict
from features import fit_edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260, 20261, 20262, 20263])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stream-base", type=int, default=0,
                    help="offset into the jitter stream family, so separate runs "
                         "see different re-encodings")
    ap.add_argument("--out", type=Path, default=Path("artifacts/v2"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw_all)
    args.out.mkdir(parents=True, exist_ok=True)

    oof_seeds = {a: [] for a in args.arms}
    test_parts = {a: [] for a in args.arms}
    started = time.time()

    for si, seed in enumerate(args.seeds):
        t0 = time.time()
        X, cats = catboost_frame(raw_all, edges, stream_offset=args.stream_base + si + 1)
        Xtr = X.iloc[: len(train)].reset_index(drop=True)
        Xte = X.iloc[len(train):].reset_index(drop=True)
        print(f"[seed {seed}] frame {X.shape} cats={len(cats)} ({time.time() - t0:.0f}s)", flush=True)

        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(Xtr, y))
        for name in args.arms:
            t1 = time.time()
            oof = np.zeros(len(y))
            for f, (ti, vi) in enumerate(folds):
                pv, pt = fit_predict(name, Xtr.iloc[ti], Xtr.iloc[vi], Xte, cats, y[ti], seed + f)
                oof[vi] = pv
                test_parts[name].append(rankdata(pt) / len(pt))
            oof_seeds[name].append(rankdata(oof) / len(oof))
            print(f"    {name:8s} oof={roc_auc_score(y, oof):.5f} ({time.time() - t1:.0f}s)", flush=True)

    summary = {"seeds": args.seeds, "folds": args.folds, "arms": {}}
    for name in args.arms:
        oof = np.mean(oof_seeds[name], axis=0)
        pred = np.mean(test_parts[name], axis=0)
        np.savez_compressed(args.out / f"arm_{name}.npz", oof=oof, test=pred, y=y)
        summary["arms"][name] = {
            "bagged_oof_auc": float(roc_auc_score(y, oof)),
            "per_seed_auc": [float(roc_auc_score(y, o)) for o in oof_seeds[name]],
        }
        print(f"[final] {name}: {summary['arms'][name]['bagged_oof_auc']:.5f}", flush=True)

    names = list(args.arms)
    summary["oof_corr"] = {
        f"{a}~{b}": float(np.corrcoef(np.mean(oof_seeds[a], axis=0), np.mean(oof_seeds[b], axis=0))[0, 1])
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    summary["elapsed_sec"] = round(time.time() - started, 1)
    (args.out / "arms_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
