#!/usr/bin/env python3
"""Ordered B5-no-xbin arm with early stopping (same protocol as champion ord_noxb_bag).

Train NEW seeds and/or hyperparameter variants; bag into a new arm for max3+.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from insurance_claim.model import TARGET  # noqa: E402
from insurance_claim.train_b5_focus import VIEWS  # noqa: E402

DATA = ROOT / "data"
ART = ROOT / "artifacts" / "beat_max3" / "train"
N_SPLITS = 5


def run_seed(builder, params, features, test, y, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    oof = np.zeros(len(features), dtype=np.float64)
    te = np.zeros(len(test), dtype=np.float64)
    for fold, (tri, vai) in enumerate(
        StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(features, y)
    ):
        Xtr = features.iloc[tri].reset_index(drop=True)
        Xva = features.iloc[vai].reset_index(drop=True)
        ytr = y.iloc[tri].reset_index(drop=True)
        yva = y.iloc[vai].reset_index(drop=True)
        tr_df, va_df, te_df, cats = builder(Xtr, Xva, test.copy())
        m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
        m.fit(tr_df, ytr, eval_set=(va_df, yva), cat_features=cats, use_best_model=True, verbose=False)
        oof[vai] = m.predict_proba(va_df)[:, 1]
        te += m.predict_proba(te_df)[:, 1] / N_SPLITS
        print(
            f"  seed={seed} fold={fold} auc={roc_auc_score(yva, oof[vai]):.5f} "
            f"best={m.get_best_iteration()}",
            flush=True,
        )
    return oof, te, float(roc_auc_score(y, oof))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--view", default="b5", choices=list(VIEWS))
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    builder, base_p = VIEWS[args.view]
    params = {k: v for k, v in base_p.items() if k != "random_seed"}
    params.pop("l2_leaf_reg", None)
    ncpu = os.cpu_count() or 4
    params.update(
        depth=args.depth,
        iterations=args.iterations,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        boosting_type="Ordered",
        thread_count=max(1, ncpu // 2),
    )

    oofs, tes, per = [], [], []
    t0 = time.time()
    for seed in args.seeds:
        part = ART / f"part_{args.tag}_s{seed}.npz"
        if part.exists():
            d = np.load(part)
            oof, te, auc = d["oof"], d["test_pred"], float(d["auc"])
            print(f"[resume] {args.tag} seed {seed}: OOF={auc:.6f}", flush=True)
        else:
            t1 = time.time()
            oof, te, auc = run_seed(builder, params, features, test, y, seed)
            np.savez(part, oof=oof, test_pred=te, auc=auc, seed=seed)
            print(f"[{args.tag}] seed {seed}: OOF={auc:.6f} ({time.time()-t1:.0f}s)", flush=True)
        oofs.append(oof)
        tes.append(te)
        per.append(auc)

    pooled_oof = np.mean(np.vstack(oofs), axis=0)
    pooled_te = np.mean(np.vstack(tes), axis=0)
    rank_oof = np.mean(np.vstack([rankdata(o) for o in oofs]), axis=0)
    rank_te = np.mean(np.vstack([rankdata(t) for t in tes]), axis=0)
    pool_auc = float(roc_auc_score(y, pooled_oof))
    rank_auc = float(roc_auc_score(y, rank_oof))
    out = ART.parent / f"{args.tag}.npz"
    np.savez(
        out,
        oof=pooled_oof,
        test_pred=pooled_te,
        oof_rankpool=rank_oof,
        test_rankpool=rank_te,
        per_seed=np.array(per),
        seeds=np.array(args.seeds),
        pool="mean",
    )
    meta = {
        "tag": args.tag,
        "seeds": args.seeds,
        "params": {
            k: params[k]
            for k in ("depth", "iterations", "learning_rate", "l2_leaf_reg", "boosting_type")
        },
        "view": args.view,
        "pooled_oof_auc": pool_auc,
        "rankpool_oof_auc": rank_auc,
        "per_seed": per,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "ES arm: OOF optimistic; test honest",
    }
    (ART.parent / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
