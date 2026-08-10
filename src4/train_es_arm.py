"""ES upgrade of merger_ord / cat_alt under the same 5-fold seed grid as max3.

Honest fixed-tree arms already exist (merger_ord8, v2_cat_alt8). This trainer
allows early stopping on the outer fold so TEST predictions can pick a better
iteration count — same rationale as ord_noxb_bag in zcode-v4-max3.

OOF will be mildly optimistic; fuse_v4max3pro.py tags these arms as `es`.
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "src2"))
from arms import CAT_BASE, ARMS, altboost_frame, catboost_frame  # noqa: E402
from features import fit_edges, fit_edges_alt  # noqa: E402

ART = ROOT / "artifacts" / "v4max3pro"
DATA = ROOT / "data"
SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)


def run_one(kind: str, seed: int, stream_offset: int, threads: int):
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)

    if kind == "merger_ord":
        edges = fit_edges(raw)
        X, cats = catboost_frame(raw, edges, stream_offset=stream_offset, n_views=4)
        depth, iterations = 5, 1200
        boosting = "Ordered"
        extra = {}
    elif kind == "cat_alt":
        edges = fit_edges_alt(raw)
        X, cats = altboost_frame(raw, edges, stream_offset=stream_offset)
        depth, iterations = ARMS["cat_alt"]["depth"], 1200
        boosting = "Plain"
        extra = {"l2_leaf_reg": 6, "one_hot_max_size": 12}
    else:
        raise ValueError(kind)

    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train) :].reset_index(drop=True)
    params = dict(CAT_BASE)
    params.update(extra)
    params.update(
        dict(
            depth=depth,
            iterations=iterations,
            boosting_type=boosting,
            thread_count=threads,
            eval_metric="AUC",
            od_type="Iter",
            od_wait=100,
        )
    )

    oof = np.zeros(len(y))
    te = np.zeros(len(test))
    fold_meta = []
    t0 = time.time()
    for fold, (ti, vi) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y)
    ):
        m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
        m.fit(
            Xtr.iloc[ti],
            y[ti],
            eval_set=(Xtr.iloc[vi], y[vi]),
            cat_features=cats,
            use_best_model=True,
            verbose=False,
        )
        oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        te += m.predict_proba(Xte)[:, 1] / 5
        fold_meta.append(
            {
                "fold": fold,
                "valid_auc": float(roc_auc_score(y[vi], oof[vi])),
                "best_iter": int(m.get_best_iteration() or -1),
            }
        )
        print(
            f"[{kind}_es] seed={seed} fold={fold} auc={fold_meta[-1]['valid_auc']:.5f} "
            f"best={fold_meta[-1]['best_iter']}",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    print(f"[{kind}_es] seed={seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_es_s{seed}"
    np.savez_compressed(ART / f"part_{tag}.npz", oof=oof, test=te, y=y)
    (ART / f"part_{tag}.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "seed": seed,
                "stream_offset": stream_offset,
                "oof_auc": auc,
                "folds_meta": fold_meta,
                "elapsed_sec": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    return auc


def merge_parts(kind: str, out_name: str):
    parts = sorted(ART.glob(f"part_{kind}_es_s*.npz"))
    if not parts:
        raise SystemExit(f"no parts for {kind}")
    oofs, tes, aucs, seeds = [], [], [], []
    y = None
    for p in parts:
        d = np.load(p)
        oofs.append(rankdata(d["oof"]) / len(d["oof"]))
        tes.append(rankdata(d["test"]) / len(d["test"]))
        y = d["y"]
        aucs.append(float(roc_auc_score(y, d["oof"])))
        seeds.append(int(p.stem.split("_s")[-1]))
    oof = np.mean(oofs, 0)
    te = np.mean(tes, 0)
    bag = float(roc_auc_score(y, oof))
    np.savez_compressed(
        ART / f"{out_name}.npz",
        oof=oof,
        test_pred=te,
        per_seed=np.array(aucs),
        seeds=np.array(seeds),
        y=y,
    )
    print(f"merged {out_name}: bag={bag:.6f} n={len(parts)} per={aucs}")
    return bag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["merger_ord", "cat_alt"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream-offset", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    out = "merger_ord_es" if args.kind == "merger_ord" else "cat_alt_es"
    if args.merge_only:
        merge_parts(args.kind, out)
        return 0
    si = SEEDS.index(args.seed) if args.seed in SEEDS else 0
    offset = args.stream_offset if args.stream_offset is not None else si + 1
    run_one(args.kind, args.seed, offset, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
