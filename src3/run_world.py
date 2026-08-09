"""Train one encoding world for one seed and save its OOF / test prediction.

Written as a one-seed worker so several seeds can run as separate single-thread
processes; on a 4-core box that finishes a batch of four roughly twice as fast
as one four-threaded run per seed.  ``src3/merge_seeds.py`` pools the parts
afterwards, which is exactly what src2/merge_runs.py does for the existing arms.

Protocol is unchanged from the rest of the branch: stratified 5-fold, fixed
iteration count, no early stopping, no look at the outer validation fold, and
feature engineering that never touches the label.
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
from worlds import fit_edges_w4, fit_edges_w5, w4_frame, w5_frame
from v5_world import fit_edges_w6, w6_frame
from v5_w7 import fit_edges_w7, w7_frame
from v5_w8 import fit_edges_w8, w8_frame
from v5_w9 import fit_edges_w9, w9_frame
from v5_w10 import fit_edges_w10, w10_frame

WORLDS = {
    "main": (fit_edges, catboost_frame),
    "alt": (fit_edges_alt, altboost_frame),
    "alt2": (fit_edges_alt2, alt2_frame),
    "w4": (fit_edges_w4, w4_frame),
    "w5": (fit_edges_w5, w5_frame),
    "w6": (fit_edges_w6, w6_frame),
    "w7": (fit_edges_w7, w7_frame),
    "w8": (fit_edges_w8, w8_frame),
    "w9": (fit_edges_w9, w9_frame),
    "w10": (fit_edges_w10, w10_frame),
}

# Depth/iterations mirror the tuned settings of the existing CatBoost arms; the
# screening sweep found no configuration outside the noise band, so the new
# worlds reuse the known-good ones rather than inventing a fresh guess.
PRESETS = {
    "d5": dict(depth=5, iterations=1000),
    "d6": dict(depth=6, iterations=700, bagging_temperature=1.0),
    "d6l6": dict(depth=6, iterations=800, l2_leaf_reg=6, one_hot_max_size=12),
}

BASE = dict(loss_function="Logloss", learning_rate=0.03, l2_leaf_reg=10,
            random_strength=0.7, verbose=False, allow_writing_files=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=sorted(WORLDS))
    ap.add_argument("--preset", default="d6l6", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream-offset", type=int, default=None,
                    help="jitter family; defaults to a function of the seed")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("artifacts/worlds"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)

    fit_e, make = WORLDS[args.world]
    offset = args.stream_offset if args.stream_offset is not None else (args.seed % 97) + 1
    t0 = time.time()
    X, cats = make(raw, fit_e(raw), stream_offset=offset)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train):].reset_index(drop=True)

    params = dict(BASE)
    params.update(PRESETS[args.preset])
    params["thread_count"] = args.threads

    oof = np.zeros(len(y))
    test_parts = []
    for f, (ti, vi) in enumerate(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(Xtr, y)
    ):
        m = CatBoostClassifier(**params, random_seed=args.seed + f)
        m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
        oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        test_parts.append(rankdata(m.predict_proba(Xte)[:, 1]) / len(Xte))

    auc = float(roc_auc_score(y, oof))
    tag = f"{args.world}_{args.preset}_s{args.seed}_f{args.folds}"
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"part_{tag}.npz",
                        oof=rankdata(oof) / len(oof),
                        test=np.mean(test_parts, axis=0), y=y)
    (args.out / f"part_{tag}.json").write_text(json.dumps(
        {"world": args.world, "preset": args.preset, "seed": args.seed,
         "folds": args.folds, "stream_offset": offset, "oof_auc": auc,
         "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
    print(f"{tag} oof={auc:.5f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
