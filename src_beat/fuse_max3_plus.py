#!/usr/bin/env python3
"""Beat-max3 fuse: keep champion 3 arms + optional new arms.

Fusion = elementwise max of per-arm rank-normalized predictions
(identical protocol to submission_v4_max3 / LB 0.71222).

Hard rules (supervisor):
- NEVER drop ord_noxb_bag
- Always include merger_ord8 + v2_cat_alt8 + ord_noxb_bag
- Gate on nested Δ vs max3, Spearman band, blocks+
- Do NOT extrapolate nested + 0.0095 → LB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("artifacts/beat_max3")
SUB = Path("submissions")
TRAIN = Path("data/train.csv")
TEST = Path("data/test.csv")
MAX3_LB = 0.71222
BASE_NAMES = ("merger_ord8", "v2_cat_alt8", "ord_noxb_bag")


def rank01(a: np.ndarray) -> np.ndarray:
    return rankdata(a.astype(np.float64)) / len(a)


def nested_auc(blend: np.ndarray, y: np.ndarray, n_blocks: int = 5) -> float:
    out = np.zeros(len(y), dtype=np.float64)
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(blend[b]) / len(b)
    return float(roc_auc_score(y, out))


def block_aucs(blend: np.ndarray, y: np.ndarray, n_blocks: int = 5) -> list[float]:
    return [
        float(roc_auc_score(y[b], blend[b]))
        for b in np.array_split(np.arange(len(y)), n_blocks)
    ]


def load_arm(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = ART / f"{name}.npz"
    if not path.exists():
        # allow arm_* aliases from v4_ext copies
        alt = ART / f"arm_{name}.npz"
        path = alt if alt.exists() else path
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=np.float64)
    if "test_pred" in d.files:
        te = np.asarray(d["test_pred"], dtype=np.float64)
    elif "test" in d.files:
        te = np.asarray(d["test"], dtype=np.float64)
    else:
        raise KeyError(f"{path} missing test_pred/test")
    return oof, te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=["plus_strong"])
    ap.add_argument("--tag", default="max3_plus")
    ap.add_argument("--min-delta", type=float, default=0.0010)
    ap.add_argument("--spearman-lo", type=float, default=0.985)
    ap.add_argument("--spearman-hi", type=float, default=0.997)
    ap.add_argument("--min-blocks-plus", type=int, default=4)
    ap.add_argument("--min-new-win", type=float, default=0.15)
    args = ap.parse_args()

    y = pd.read_csv(TRAIN, usecols=["label"])["label"].to_numpy(np.int8)
    tid = pd.read_csv(TEST, usecols=["id"])["id"].to_numpy()

    names = list(BASE_NAMES) + list(args.extra)
    missing = [n for n in names if not (ART / f"{n}.npz").exists()]
    if missing:
        raise SystemExit(f"missing arms under {ART}: {missing}")

    raw = {n: load_arm(n) for n in names}
    oofs = {n: rank01(raw[n][0]) for n in names}
    tests = {n: rank01(raw[n][1]) for n in names}

    base_oof = np.maximum.reduce([oofs[n] for n in BASE_NAMES])
    base_test = np.maximum.reduce([tests[n] for n in BASE_NAMES])
    base_nested = nested_auc(base_oof, y)
    base_blocks = block_aucs(base_oof, y)

    cand_oof = np.maximum.reduce([oofs[n] for n in names])
    cand_test = np.maximum.reduce([tests[n] for n in names])
    cand_nested = nested_auc(cand_oof, y)
    cand_blocks = block_aucs(cand_oof, y)
    delta = cand_nested - base_nested
    sp = float(spearmanr(cand_test, base_test).correlation)
    blocks_plus = int(sum(c > b for c, b in zip(cand_blocks, base_blocks)))

    new_win = {n: float(np.mean(oofs[n] > base_oof)) for n in args.extra}
    arm_auc = {n: float(roc_auc_score(y, raw[n][0])) for n in names}

    gate = {
        "G1_spearman": args.spearman_lo <= sp <= args.spearman_hi,
        "G2_delta": delta >= args.min_delta,
        "G3_blocks_plus": blocks_plus >= args.min_blocks_plus,
        "G4_keep_noxb": "ord_noxb_bag" in names,
        "G5_new_win": all(v >= args.min_new_win for v in new_win.values()) if new_win else True,
        "G6_keep_base": all(n in names for n in BASE_NAMES),
    }
    passed = all(gate.values())

    SUB.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    out_csv = SUB / f"submission_{args.tag}.csv"
    frame = pd.DataFrame({"id": tid, "label": cand_test.astype(np.float64)})
    frame.to_csv(out_csv, index=False)
    frame.to_csv(ART / f"submission_{args.tag}.csv", index=False)

    report = {
        "tag": args.tag,
        "arms": names,
        "fusion": "max(rank)",
        "base_nested": base_nested,
        "cand_nested": cand_nested,
        "delta": delta,
        "spearman_vs_max3": sp,
        "blocks_plus": f"{blocks_plus}/5",
        "base_blocks": base_blocks,
        "cand_blocks": cand_blocks,
        "new_arm_win_rate": new_win,
        "arm_oof_auc": arm_auc,
        "gate": gate,
        "passed": passed,
        "max3_lb_anchor": MAX3_LB,
        "caution": (
            "Nested is optimistic when ES arms are present. "
            "Do NOT add +0.0095 gap. Expect LB ≈ max3 + fraction of delta."
        ),
        "submission": str(out_csv),
    }
    (ART / f"report_{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASSED" if passed else "FAILED_GATE")


if __name__ == "__main__":
    main()
