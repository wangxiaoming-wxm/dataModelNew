"""Pool V4 per-seed parts into one arm file.

V4 part names include the loss tag written by ``src4/run_world.py``:
``part_<world>_<preset>_<loss>_s<seed>_f<folds>.npz``.  This helper mirrors
the V3 rank-average merge while keeping the V4 artifact path self-contained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/worlds20"))
    ap.add_argument("--world", required=True)
    ap.add_argument("--preset", default="d6l6")
    ap.add_argument("--loss", default="logloss")
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--name", default=None, help="arm name to write (default cat_<world>)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/v4"))
    args = ap.parse_args()

    parts = sorted(
        args.dir.glob(f"part_{args.world}_{args.preset}_{args.loss}_s*_f{args.folds}.npz")
    )
    if not parts:
        raise SystemExit(
            f"no parts for {args.world}/{args.preset}/{args.loss}/f{args.folds} in {args.dir}"
        )

    oofs, tests, y = [], [], None
    for p in parts:
        z = np.load(p)
        oofs.append(z["oof"])
        tests.append(z["test"])
        if y is None:
            y = z["y"]
        elif not np.array_equal(y, z["y"]):
            raise SystemExit(f"label mismatch in {p}")

    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    name = args.name or f"cat_{args.world}"
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"arm_{name}.npz", oof=oof, test=test, y=y)

    info = {
        "arm": name,
        "world": args.world,
        "preset": args.preset,
        "loss": args.loss,
        "folds": args.folds,
        "n_seeds": len(parts),
        "parts": [p.name for p in parts],
        "per_seed_auc": [float(roc_auc_score(y, o)) for o in oofs],
        "bagged_oof_auc": float(roc_auc_score(y, oof)),
    }
    (args.out / f"arm_{name}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
