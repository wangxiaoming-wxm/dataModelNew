#!/usr/bin/env python3
"""Refresh stage_best: bag available new noxb parts + screen top combos under gates.

Writes submissions/submission_max3_stage_best.csv when a candidate beats current best gate delta.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "beat_max3"
TRAIN_DIR = ART / "train"
SUB = ROOT / "submissions"
BASE = ("merger_ord8", "v2_cat_alt8", "ord_noxb_bag")
CORE_EXTRA = ["plus_strong", "noxb10", "cat_w12_d5"]


def nested_auc(oof: np.ndarray, y: np.ndarray) -> float:
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), 5):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def load_arm(name: str) -> tuple[np.ndarray, np.ndarray]:
    p = ART / f"{name}.npz"
    d = np.load(p, allow_pickle=True)
    oof = np.asarray(d["oof"], float)
    te = np.asarray(d["test_pred"] if "test_pred" in d.files else d["test"], float)
    return rankdata(oof) / len(oof), rankdata(te) / len(te)


def bag_parts(prefix: str, out_name: str) -> str | None:
    parts = sorted(TRAIN_DIR.glob(f"part_{prefix}_s*.npz"))
    if len(parts) < 2:
        return None
    oofs = [np.load(p)["oof"] for p in parts]
    tes = [np.load(p)["test_pred"] for p in parts]
    o = np.mean(oofs, 0)
    t = np.mean(tes, 0)
    np.savez(ART / f"{out_name}.npz", oof=o, test_pred=t, n_parts=len(parts))
    return out_name


def eval_combo(names: list[str], base_o, base_t, y, tid) -> dict:
    oo = [base_o] + [load_arm(n)[0] for n in names]
    tt = [base_t] + [load_arm(n)[1] for n in names]
    fo = np.maximum.reduce(oo)
    ft = np.maximum.reduce(tt)
    nest = nested_auc(fo, y)
    base_n = nested_auc(base_o, y)
    bp = sum(
        roc_auc_score(y[b], fo[b]) > roc_auc_score(y[b], base_o[b])
        for b in np.array_split(np.arange(len(y)), 5)
    )
    sp = float(spearmanr(ft, base_t).correlation)
    return {
        "extra": names,
        "nested": nest,
        "delta": nest - base_n,
        "sp": sp,
        "blocks_plus": bp,
        "te": ft,
        "passed": (nest - base_n) >= 0.001 and 0.985 <= sp <= 0.997 and bp >= 4,
    }


def main() -> None:
    y = pd.read_csv(ROOT / "data/train.csv")["label"].astype(int).values
    tid = pd.read_csv(ROOT / "data/test.csv")["id"]
    base_o = np.maximum.reduce([load_arm(n)[0] for n in BASE])
    base_t = np.maximum.reduce([load_arm(n)[1] for n in BASE])

    # refresh interim bags
    dynamic = []
    for prefix, out in [
        ("ord_noxb_new16", "ord_noxb_new_interim"),
        ("ord_noxb_d8x16", "ord_noxb_d8_interim"),
        ("ord_noxb_slow7", "ord_noxb_slow_interim"),
        ("ord_noxb_b1x16", "ord_noxb_b1_interim"),
        ("plus_hq8", "plus_hq_interim"),
        ("cofeh_hq8", "cofeh_interim"),
        ("goldmine_hq8", "goldmine_interim"),
    ]:
        name = bag_parts(prefix, out)
        if name:
            dynamic.append(name)
        # also register finished full arms
        if (ART / f"{prefix}.npz").exists():
            dynamic.append(prefix)

    # available static extras
    static = [n for n in CORE_EXTRA + ["semantic_rmse", "plus_v10", "ordered_bag"] if (ART / f"{n}.npz").exists()]
    pool = sorted(set(static + dynamic))

    # candidates: core trio + optional 1-2 dynamic arms
    cands = []
    cands.append(eval_combo(CORE_EXTRA, base_o, base_t, y, tid))
    for d in dynamic:
        cands.append(eval_combo(CORE_EXTRA + [d], base_o, base_t, y, tid))
        cands.append(eval_combo(["plus_strong", d], base_o, base_t, y, tid))
    for a, b in combinations([x for x in dynamic if x], 2):
        cands.append(eval_combo(CORE_EXTRA + [a, b], base_o, base_t, y, tid))

    cands = [c for c in cands if c["passed"]]
    cands.sort(key=lambda c: -c["delta"])
    report = {
        "n_pool": len(pool),
        "pool": pool,
        "top": [{k: v for k, v in c.items() if k != "te"} for c in cands[:15]],
    }
    (ART / "stage_best_screen.json").write_text(json.dumps(report, indent=2))
    if not cands:
        print("NO_PASSING_CANDIDATE")
        return
    best = cands[0]
    tag = "max3_stage_best"
    pd.DataFrame({"id": tid, "label": best["te"]}).to_csv(SUB / f"submission_{tag}.csv", index=False)
    pd.DataFrame({"id": tid, "label": best["te"]}).to_csv(ART / f"submission_{tag}.csv", index=False)
    meta = {k: v for k, v in best.items() if k != "te"}
    meta["tag"] = tag
    (ART / f"report_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"shipped": tag, **meta}, indent=2))


if __name__ == "__main__":
    main()
