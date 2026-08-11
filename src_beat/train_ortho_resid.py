#!/usr/bin/env python3
"""Orthogonal residual arm: emphasize regions where max3 under-ranks positives.

Hard gate (same as strategy probes):
  pooled OOF AUC > 0.690 AND Spearman vs merger_ord8 < 0.88
If fail → do not enter max fusion.
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
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from insurance_claim.train_b5_focus import VIEWS  # noqa: E402

DATA = ROOT / "data"
ART = ROOT / "artifacts" / "beat_max3"
OUT = ART / "probes"


def rk(a):
    return rankdata(np.asarray(a, float)) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ortho_hard")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3100, 3101])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int)
    feats = train.drop(columns=["label"])
    mo = np.load(ART / "merger_ord8.npz")["oof"]
    ca = np.load(ART / "v2_cat_alt8.npz")["oof"]
    od = np.load(ART / "ord_noxb_bag.npz")["oof"]
    fused = np.maximum.reduce([rk(mo), rk(ca), rk(od)])

    # sample weight: boost positives that max3 ranks low + negatives ranked high
    # (hard mistakes). Clip to keep training stable.
    w = np.ones(len(y), dtype=float)
    pos = y.to_numpy() == 1
    neg = ~pos
    # lower fused → harder positive
    w[pos] = 1.0 + 3.0 * (1.0 - fused[pos])
    w[neg] = 1.0 + 2.0 * fused[neg]
    w = np.clip(w, 1.0, 5.0)

    ncpu = os.cpu_count() or 4
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=15,
        boosting_type="Ordered",
        od_type="Iter",
        od_wait=150,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, ncpu // 3),
    )
    builder, _ = VIEWS["b5"]

    oofs, tes, per = [], [], []
    t0 = time.time()
    for seed in args.seeds:
        part = OUT / f"part_{args.tag}_s{seed}.npz"
        if part.exists():
            d = np.load(part)
            oof, te_p, auc = d["oof"], d["test_pred"], float(d["auc"])
            print(f"[resume] {args.tag} s{seed} {auc:.5f}", flush=True)
        else:
            oof = np.zeros(len(train))
            te_p = np.zeros(len(test))
            for fold, (a, b) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(feats, y)):
                Xtr = feats.iloc[a].reset_index(drop=True)
                Xva = feats.iloc[b].reset_index(drop=True)
                ytr = y.iloc[a].reset_index(drop=True)
                yva = y.iloc[b].reset_index(drop=True)
                wtr = w[a]
                tr, va, te, cats = builder(Xtr, Xva, test.copy())
                m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
                m.fit(tr, ytr, sample_weight=wtr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
                oof[b] = m.predict_proba(va)[:, 1]
                te_p += m.predict_proba(te)[:, 1] / args.folds
                print(
                    f"  {args.tag} s{seed} f{fold} auc={roc_auc_score(yva, oof[b]):.5f} best={m.get_best_iteration()}",
                    flush=True,
                )
            auc = float(roc_auc_score(y, oof))
            np.savez(part, oof=oof, test_pred=te_p, auc=auc)
            print(f"[{args.tag}] seed {seed} OOF={auc:.5f}", flush=True)
        oofs.append(oof)
        tes.append(te_p)
        per.append(auc)

    oof = np.mean(np.vstack(oofs), 0)
    te = np.mean(np.vstack(tes), 0)
    auc = float(roc_auc_score(y, oof))
    corr_mo = float(spearmanr(oof, mo).correlation)
    corr_ca = float(spearmanr(oof, ca).correlation)
    gate = {
        "auc_gt_0.690": auc > 0.690,
        "corr_mo8_lt_0.88": corr_mo < 0.88,
        "corr_ca8_lt_0.88": corr_ca < 0.88,
    }
    admit = all(gate.values())
    np.savez(ART / f"probe_{args.tag}.npz", oof=oof, test_pred=te, per_seed=np.array(per), seeds=np.array(args.seeds))
    report = {
        "mode": args.tag,
        "pooled_oof_auc": auc,
        "per_seed": per,
        "corr_spearman_mo8": corr_mo,
        "corr_spearman_ca8": corr_ca,
        "gate": gate,
        "admit_to_max": admit,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "hard-example reweight; refuse max if gate fails",
    }
    (OUT / f"report_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    # symlink-style name for build_ship maybe_admit_probe (expects exp*)
    if admit:
        np.savez(ART / "probe_exp_ortho.npz", oof=oof, test_pred=te)
        (OUT / "report_exp_ortho.json").write_text(json.dumps({**report, "admit_to_max": True}, indent=2))


if __name__ == "__main__":
    main()
