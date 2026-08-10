"""V4 single-seed world trainer.

Extends src3/run_world.py with:
  * worlds w6 / w7
  * optional fold counts (default 10; V4 probes use 15/20)
  * optional loss_function override (must NOT introduce eval_set / early stop)

Protocol unchanged: fixed iterations, no validation-fold peeking, label-free FE.
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
from worlds_v4 import (
    fit_edges_w6,
    fit_edges_w7,
    fit_edges_w8,
    fit_edges_w9,
    fit_edges_w10,
    fit_edges_w11,
    w6_frame,
    w7_frame,
    w8_frame,
    w9_frame,
    w10_frame,
    w11_frame,
)

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
    "w11": (fit_edges_w11, w11_frame),
}

PRESETS = {
    "d5": dict(depth=5, iterations=1000),
    "d6": dict(depth=6, iterations=700, bagging_temperature=1.0),
    "d6l6": dict(depth=6, iterations=800, l2_leaf_reg=6, one_hot_max_size=12),
    # slightly more trees when each fold sees more data (20-fold); still fixed
    "d5x": dict(depth=5, iterations=1200),
    "d6x": dict(depth=6, iterations=900, bagging_temperature=1.0),
    # screening only — never used for headline numbers
    "fast": dict(depth=5, iterations=400),
}

BASE = dict(
    loss_function="Logloss",
    learning_rate=0.03,
    l2_leaf_reg=10,
    random_strength=0.7,
    verbose=False,
    allow_writing_files=False,
)


def stratified_subsample(idx: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return idx
    if not 0.0 < frac < 1.0:
        raise ValueError(f"train fraction must be in (0, 1], got {frac}")
    rng = np.random.default_rng(seed)
    kept = []
    for cls in np.unique(y[idx]):
        cls_idx = idx[y[idx] == cls]
        n_keep = max(1, int(round(len(cls_idx) * frac)))
        kept.append(rng.choice(cls_idx, size=n_keep, replace=False))
    out = np.concatenate(kept)
    rng.shuffle(out)
    return out


def ratio_mid_focus_upsample(
    idx: np.ndarray,
    ratio: np.ndarray,
    lo: float,
    hi: float,
    factor: float,
    seed: int,
) -> np.ndarray:
    """Repeat mid-ratio training rows. Cuts and membership are label-free."""
    if factor <= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    mid = idx[(ratio[idx] >= lo) & (ratio[idx] <= hi)]
    if len(mid) == 0:
        return idx
    n_extra = int(round(len(mid) * (factor - 1.0)))
    extra = rng.choice(mid, size=n_extra, replace=True)
    out = np.concatenate([idx, extra])
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=sorted(WORLDS))
    ap.add_argument("--preset", default="d6l6", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream-offset", type=int, default=None)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--loss", default="Logloss",
                    help="CatBoost loss_function; no eval_set is ever passed")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="Fixed stratified fraction of each training fold to fit")
    ap.add_argument(
        "--iter-jitter",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Per-fold fixed iteration draw from [LO, HI] using seed+fold only "
             "(no eval_set / early stop). Replaces preset iterations.",
    )
    ap.add_argument(
        "--ratio-mid-focus",
        type=float,
        default=1.0,
        help="Label-free upsample factor for training rows whose source-median "
             "ratio sits in the global central 50%% (edges from train+test).",
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/v4_probe"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)

    # label-free ratio for optional mid-band upsampling
    med = raw.groupby("source")["condition"].transform("median")
    cond_r = (pd.to_numeric(raw["condition"]) / med.replace(0, np.nan)).fillna(1.0)
    ratio_all = (pd.to_numeric(raw["days"]) / cond_r.clip(lower=1e-9)).to_numpy(dtype=float)
    ratio_tr = ratio_all[: len(train)]
    q25, q75 = np.quantile(ratio_all, [0.25, 0.75])

    fit_e, make = WORLDS[args.world]
    offset = args.stream_offset if args.stream_offset is not None else (args.seed % 97) + 1
    t0 = time.time()
    X, cats = make(raw, fit_e(raw), stream_offset=offset)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train) :].reset_index(drop=True)

    params = dict(BASE)
    params.update(PRESETS[args.preset])
    params["thread_count"] = args.threads
    params["loss_function"] = args.loss
    if args.iter_jitter is not None:
        lo, hi = int(args.iter_jitter[0]), int(args.iter_jitter[1])
        if not (1 <= lo <= hi):
            raise SystemExit(f"invalid --iter-jitter range [{lo}, {hi}]")

    oof = np.zeros(len(y))
    test_parts = []
    fold_iters = []
    for f, (ti, vi) in enumerate(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(Xtr, y)
    ):
        fold_params = dict(params)
        if args.iter_jitter is not None:
            lo, hi = int(args.iter_jitter[0]), int(args.iter_jitter[1])
            # deterministic draw from seed+fold only — never looks at held-out labels
            rng = np.random.default_rng(args.seed * 1_000_003 + f * 97 + 13)
            fold_params["iterations"] = int(rng.integers(lo, hi + 1))
            fold_iters.append(fold_params["iterations"])
        fit_idx = stratified_subsample(ti, y, args.train_frac, seed=args.seed * 1000 + f)
        fit_idx = ratio_mid_focus_upsample(
            fit_idx, ratio_tr, float(q25), float(q75),
            factor=float(args.ratio_mid_focus), seed=args.seed * 1000 + f + 17,
        )
        m = CatBoostClassifier(**fold_params, random_seed=args.seed + f)
        m.fit(Xtr.iloc[fit_idx], y[fit_idx], cat_features=cats, verbose=False)
        oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        test_parts.append(rankdata(m.predict_proba(Xte)[:, 1]) / len(Xte))

    auc = float(roc_auc_score(y, oof))
    loss_tag = args.loss.lower().replace(":", "")
    frac_tag = "" if args.train_frac == 1.0 else f"_sf{int(round(args.train_frac * 100)):02d}"
    rit_tag = ""
    if args.iter_jitter is not None:
        rit_tag = f"_rit{int(args.iter_jitter[0])}_{int(args.iter_jitter[1])}"
    mid_tag = ""
    if float(args.ratio_mid_focus) > 1.0:
        mid_tag = f"_mid{int(round(float(args.ratio_mid_focus) * 10)):02d}"
    tag = f"{args.world}_{args.preset}_{loss_tag}{frac_tag}{rit_tag}{mid_tag}_s{args.seed}_f{args.folds}"
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"part_{tag}.npz",
        oof=rankdata(oof) / len(oof),
        test=np.mean(test_parts, axis=0),
        y=y,
    )
    meta = {
        "world": args.world,
        "preset": args.preset,
        "seed": args.seed,
        "folds": args.folds,
        "loss": args.loss,
        "train_frac": args.train_frac,
        "iter_jitter": list(args.iter_jitter) if args.iter_jitter is not None else None,
        "ratio_mid_focus": float(args.ratio_mid_focus),
        "ratio_mid_bounds": [float(q25), float(q75)],
        "fold_iterations": fold_iters,
        "stream_offset": offset,
        "oof_auc": auc,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / f"part_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"{tag} oof={auc:.5f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
