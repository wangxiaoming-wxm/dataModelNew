"""V5 gate: does w6 match the strength of V2's arms under the SAME 5-fold protocol?

A new world is admitted into ``max`` only if its bagged OOF clears 0.694.
Weaker arms have already been shown to hurt ``views_max`` (cat_alt2).

This script trains w6 on the V2-identical protocol (5-fold, fixed trees, no
early stopping) and prints the number next to the frozen V2 arm scores.  It
does not look at the public leaderboard and does not change any V2 artefact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/v5_w6"))
    ap.add_argument("--v2", type=Path, default=Path("artifacts/v2"))
    ap.add_argument("--min-bag", type=float, default=0.6940,
                    help="admission threshold into max-fusion")
    args = ap.parse_args()

    parts = sorted(args.dir.glob("part_w6_*_f5.npz"))
    if not parts:
        raise SystemExit(f"no w6 5-fold parts under {args.dir}")
    y = np.load(parts[0])["y"]
    oofs = [np.load(p)["oof"] for p in parts]
    bag = np.mean(oofs, axis=0)
    per = [float(roc_auc_score(y, o)) for o in oofs]
    bag_auc = float(roc_auc_score(y, bag))

    v2 = {}
    for a in ("cat_d5", "cat_d6", "cat_alt", "cat_alt2", "gap"):
        p = args.v2 / f"arm_{a}.npz"
        if p.exists():
            z = np.load(p)
            v2[a] = float(roc_auc_score(z["y"], z["oof"]))

    # rank corr vs each V2 arm
    br = rankdata(bag) / len(bag)
    corr = {
        a: float(np.corrcoef(br, rankdata(np.load(args.v2 / f"arm_{a}.npz")["oof"]) / len(bag))[0, 1])
        for a in v2
    }

    admit = bag_auc >= args.min_bag
    report = {
        "n_seeds": len(parts),
        "per_seed_auc": per,
        "bagged_oof_auc": bag_auc,
        "v2_arm_oof": v2,
        "rank_corr_vs_v2": corr,
        "admission_threshold": args.min_bag,
        "admitted_to_max": admit,
        "reason": (
            f"bag {bag_auc:.5f} {'≥' if admit else '<'} {args.min_bag:.5f}: "
            + ("admit into views4_max" if admit else "REJECT — would drag max down")
        ),
    }
    args.dir.mkdir(parents=True, exist_ok=True)
    (args.dir / "gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if admit else 2


if __name__ == "__main__":
    raise SystemExit(main())
