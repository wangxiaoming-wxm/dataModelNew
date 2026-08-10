"""Compute out-of-fold and test predictions for every arm on shared partitions.

Loops seeds on the outside so the (expensive) feature frame is built once per
seed and reused by every arm; all arms therefore see byte-identical folds.

When --save-raw is set, each arm also stores per-seed probabilities, ranks and
fold ids so later experiments can compare aggregators without retraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from arms import ARMS, alt2_frame, altboost_frame, catboost_frame, fit_predict
from features import fit_edges, fit_edges_alt, fit_edges_alt2


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260, 20261, 20262, 20263])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--view", choices=["main", "alt", "alt2"], default="main")
    ap.add_argument("--stream-base", type=int, default=0,
                    help="offset into the jitter stream family, so separate runs "
                         "see different re-encodings")
    ap.add_argument("--save-raw", action="store_true",
                    help="also store per-seed / per-fold raw probabilities")
    ap.add_argument("--experiment-id", type=str, default="")
    ap.add_argument("--out", type=Path, default=Path("artifacts/v2"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edge_fit = {"main": fit_edges, "alt": fit_edges_alt, "alt2": fit_edges_alt2}[args.view]
    make_frame = {"main": catboost_frame, "alt": altboost_frame, "alt2": alt2_frame}[args.view]
    edges = edge_fit(raw_all)
    args.out.mkdir(parents=True, exist_ok=True)

    oof_rank_seeds = {a: [] for a in args.arms}
    oof_prob_seeds = {a: [] for a in args.arms}
    test_rank_parts = {a: [] for a in args.arms}
    test_prob_parts = {a: [] for a in args.arms}
    fold_id_by_seed = {a: [] for a in args.arms}
    started = time.time()

    for si, seed in enumerate(args.seeds):
        t0 = time.time()
        X, cats = make_frame(raw_all, edges, stream_offset=args.stream_base + si + 1)
        Xtr = X.iloc[: len(train)].reset_index(drop=True)
        Xte = X.iloc[len(train):].reset_index(drop=True)
        print(f"[seed {seed}] frame {X.shape} cats={len(cats)} ({time.time() - t0:.0f}s)", flush=True)

        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(Xtr, y))
        for name in args.arms:
            t1 = time.time()
            oof = np.zeros(len(y))
            fold_ids = np.full(len(y), -1, dtype=np.int16)
            seed_test_probs = []
            seed_test_ranks = []
            for f, (ti, vi) in enumerate(folds):
                pv, pt = fit_predict(name, Xtr.iloc[ti], Xtr.iloc[vi], Xte, cats, y[ti], seed + f)
                oof[vi] = pv
                fold_ids[vi] = f
                seed_test_probs.append(pt)
                seed_test_ranks.append(rankdata(pt) / (len(pt) + 1.0))
                test_rank_parts[name].append(rankdata(pt) / (len(pt) + 1.0))
            oof_prob_seeds[name].append(oof.copy())
            oof_rank_seeds[name].append(rankdata(oof) / (len(oof) + 1.0))
            fold_id_by_seed[name].append(fold_ids)
            if args.save_raw:
                test_prob_parts[name].append(np.stack(seed_test_probs, axis=0))
            print(f"    {name:8s} oof={roc_auc_score(y, oof):.5f} ({time.time() - t1:.0f}s)", flush=True)

    summary = {
        "experiment_id": args.experiment_id or args.out.name,
        "git_commit": _git_commit(),
        "view": args.view,
        "seeds": args.seeds,
        "folds": args.folds,
        "stream_base": args.stream_base,
        "save_raw": bool(args.save_raw),
        "train_sha256": _sha256(Path("data/train.csv")),
        "test_sha256": _sha256(Path("data/test.csv")),
        "arms": {},
    }
    for name in args.arms:
        oof = np.mean(oof_rank_seeds[name], axis=0)
        pred = np.mean(test_rank_parts[name], axis=0)
        payload = {"oof": oof, "test": pred, "y": y}
        if args.save_raw:
            payload.update({
                "oof_prob_by_seed": np.asarray(oof_prob_seeds[name]),
                "oof_rank_by_seed": np.asarray(oof_rank_seeds[name]),
                "test_prob_by_model": np.asarray(test_prob_parts[name]),
                "fold_id_by_seed": np.asarray(fold_id_by_seed[name]),
                "seeds": np.asarray(args.seeds),
                "folds": np.asarray([args.folds]),
            })
        np.savez_compressed(args.out / f"arm_{name}.npz", **payload)
        summary["arms"][name] = {
            "bagged_oof_auc": float(roc_auc_score(y, oof)),
            "per_seed_auc": [float(roc_auc_score(y, o)) for o in oof_prob_seeds[name]],
            "per_seed_rank_auc": [float(roc_auc_score(y, o)) for o in oof_rank_seeds[name]],
        }
        print(f"[final] {name}: {summary['arms'][name]['bagged_oof_auc']:.5f}", flush=True)

    names = list(args.arms)
    summary["oof_corr"] = {
        f"{a}~{b}": float(np.corrcoef(np.mean(oof_rank_seeds[a], axis=0),
                                      np.mean(oof_rank_seeds[b], axis=0))[0, 1])
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    summary["elapsed_sec"] = round(time.time() - started, 1)
    (args.out / "arms_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
