"""Pool per-seed parts from run_world.py into one arm file.

Rank-average over seeds for the OOF and over (seed, fold) for the test
prediction -- the same pooling src2 uses, so the fused numbers stay comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/worlds"))
    ap.add_argument("--world", required=True)
    ap.add_argument("--preset", default="d6l6")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--name", default=None, help="arm name to write (default cat_<world>)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = sorted(args.dir.glob(f"part_{args.world}_{args.preset}_s*_f{args.folds}.npz"))
    if not parts:
        raise SystemExit(f"no parts for {args.world}/{args.preset}/f{args.folds} in {args.dir}")
    oofs, tests, y = [], [], None
    for p in parts:
        z = np.load(p)
        oofs.append(z["oof"])
        tests.append(z["test"])
        y = z["y"] if y is None else y

    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    name = args.name or f"cat_{args.world}"
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"arm_{name}.npz", oof=oof, test=test, y=y)

    info = {"arm": name, "world": args.world, "preset": args.preset,
            "n_seeds": len(parts), "parts": [p.name for p in parts],
            "per_seed_auc": [float(roc_auc_score(y, o)) for o in oofs],
            "bagged_oof_auc": float(roc_auc_score(y, oof))}
    (args.out / f"arm_{name}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
