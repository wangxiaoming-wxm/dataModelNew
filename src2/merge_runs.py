"""Average several independent run_oof outputs into a single arm set.

Every run contributes the same number of models, so a plain mean over the
rank-normalised arrays is exactly the pooled bag. When raw per-seed arrays are
present they are concatenated along the seed axis.
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
        loaded = []
        for d in args.inputs:
            f = d / f"arm_{name}.npz"
            if not f.exists():
                print(f"skip {name}: missing in {d}")
                break
            loaded.append(np.load(f))
        else:
            y = loaded[0]["y"]
            oof = np.mean([z["oof"] for z in loaded], axis=0)
            test = np.mean([z["test"] for z in loaded], axis=0)
            payload = {"oof": oof, "test": test, "y": y}
            if all("oof_prob_by_seed" in z.files for z in loaded):
                payload["oof_prob_by_seed"] = np.concatenate(
                    [z["oof_prob_by_seed"] for z in loaded], axis=0)
                payload["oof_rank_by_seed"] = np.concatenate(
                    [z["oof_rank_by_seed"] for z in loaded], axis=0)
                payload["fold_id_by_seed"] = np.concatenate(
                    [z["fold_id_by_seed"] for z in loaded], axis=0)
                payload["test_prob_by_model"] = np.concatenate(
                    [z["test_prob_by_model"] for z in loaded], axis=0)
                if all("seeds" in z.files for z in loaded):
                    payload["seeds"] = np.concatenate([z["seeds"] for z in loaded], axis=0)
            np.savez_compressed(args.out / f"arm_{name}.npz", **payload)
            print(f"{name}: {len(loaded)} runs -> bagged OOF AUC {roc_auc_score(y, oof):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
