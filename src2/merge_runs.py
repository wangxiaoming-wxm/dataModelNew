"""Average several independent run_oof outputs into a single arm set.

Every run contributes the same number of models, so a plain mean over the
rank-normalised arrays is exactly the pooled bag.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = sorted({p.stem.removeprefix("arm_") for d in args.inputs for p in d.glob("arm_*.npz")})
    for name in names:
        oofs, tests, y = [], [], None
        for d in args.inputs:
            f = d / f"arm_{name}.npz"
            if not f.exists():
                print(f"skip {name}: missing in {d}")
                break
            z = np.load(f)
            oofs.append(z["oof"])
            tests.append(z["test"])
            y = z["y"]
        else:
            oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
            np.savez_compressed(args.out / f"arm_{name}.npz", oof=oof, test=test, y=y)
            print(f"{name}: {len(oofs)} runs -> bagged OOF AUC {roc_auc_score(y, oof):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
