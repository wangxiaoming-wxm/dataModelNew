"""Merge seed parts produced by src4 trainers into bagged arm npz files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ART = Path(__file__).resolve().parents[1] / "artifacts" / "v4max3pro"


def merge(pattern: str, out_name: str, pool: str = "rank"):
    parts = sorted(ART.glob(pattern))
    if not parts:
        raise SystemExit(f"no parts matching {pattern}")
    oofs, tes, aucs, seeds = [], [], [], []
    y = None
    for p in parts:
        d = np.load(p)
        y = d["y"] if "y" in d else y
        o = np.asarray(d["oof"], float)
        t = np.asarray(d["test"] if "test" in d else d["test_pred"], float)
        aucs.append(float(roc_auc_score(y, o)))
        if pool == "rank":
            oofs.append(rankdata(o) / len(o))
            tes.append(rankdata(t) / len(t))
        else:
            oofs.append(o)
            tes.append(t)
        # parse seed from filename
        for tok in p.stem.split("_"):
            if tok.startswith("s") and tok[1:].isdigit():
                seeds.append(int(tok[1:]))
                break
    oof = np.mean(oofs, 0)
    te = np.mean(tes, 0)
    bag = float(roc_auc_score(y, oof if pool == "prob" else oof))
    # if rank-pooled, oof already ranks; auc on mean of ranks is fine
    np.savez_compressed(
        ART / f"{out_name}.npz",
        oof=oof,
        test_pred=te,
        per_seed=np.array(aucs),
        seeds=np.array(seeds),
        y=y,
        pool=pool,
    )
    print(f"{out_name}: n={len(parts)} bag_auc={bag:.6f} per={['%.5f'%a for a in aucs]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", choices=["rank", "prob"], default="rank")
    args = ap.parse_args()
    merge(args.pattern, args.out, args.pool)


if __name__ == "__main__":
    main()
