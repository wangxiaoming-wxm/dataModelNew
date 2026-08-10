#!/usr/bin/env python3
"""Independent audit: rebuild every candidate submission from its own arm npz,
score all fusion OOFs on one fixed ruler (5-block nested AUC), and compare each
test vector against the LB-anchored max3 submission.

Nothing here calls the delivered fuse scripts; the fusion rule is re-implemented
from the documented recipe (rank -> elementwise max -> optional clip).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

OPUS = Path("/tmp/cmp_opus/20260810-cursor-opus5-4个")
V4 = Path("/tmp/cmp_v4/tree")
DATA = V4 / "data"
N_BLOCKS = 5

Y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
TEST_ID = pd.read_csv(DATA / "test.csv")["id"].values


def rank01(a):
    a = np.asarray(a, dtype=float)
    return rankdata(a) / len(a)


def nested_auc(oof, y, n_blocks=N_BLOCKS):
    """Re-rank inside each contiguous np.array_split block, then AUC."""
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def load_arm(path: Path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    key = "test_pred" if "test_pred" in d.files else "test"
    te = np.asarray(d[key], dtype=float)
    y_stored = np.asarray(d["y"], dtype=int) if "y" in d.files else None
    return oof, te, y_stored, d.files


# candidate -> (package dir, artifact dir, [arm files], clip?, committed csv)
CAND = {
    "opus/v4_honest": dict(
        art=OPUS / "v4_honest_zcode/artifacts",
        arms=["merger_ord8", "v2_cat_alt8"],
        clip=True,
        csv=OPUS / "v4_honest_zcode/submissions/submission_v4_honest.csv",
    ),
    "opus/v4_max3": dict(
        art=OPUS / "v4_max3_zcode/artifacts",
        arms=["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"],
        clip=True,
        csv=OPUS / "v4_max3_zcode/submissions/submission_v4_max3.csv",
    ),
    "opus/v5_honest": dict(
        art=OPUS / "v5_honest/artifacts",
        arms=["merger_ord8", "v2_cat_alt8", "arm_gap"],
        clip=True,
        csv=OPUS / "v5_honest/submissions/submission_v5_honest.csv",
    ),
    "opus/v6_final": dict(
        art=OPUS / "v6_zcode/artifacts",
        arms=["merger_ord8", "v2_cat_alt8", "v6_b5v2_8raw"],
        clip=False,
        csv=OPUS / "v6_zcode/submissions/submission_v6_final.csv",
    ),
    "repo/v4_max3": dict(
        art=V4 / "artifacts/v4max3",
        arms=["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"],
        clip=True,
        csv=V4 / "submissions/submission_v4_max3.csv",
    ),
    "repo/v4max3pro": dict(
        art=None,
        arms=[
            (V4 / "artifacts/v4max3/merger_ord8.npz"),
            (V4 / "artifacts/v4max3/v2_cat_alt8.npz"),
            (V4 / "artifacts/v4max3/ord_noxb_bag.npz"),
            (V4 / "artifacts/v4max3pro/plus_strong.npz"),
            (V4 / "artifacts/v4max3pro/noxb10.npz"),
        ],
        clip=True,
        csv=V4 / "submissions/submission_v4max3pro.csv",
    ),
}


def main():
    results = {}
    fusion_te = {}
    fusion_oof = {}

    for name, cfg in CAND.items():
        paths = [
            (cfg["art"] / f"{a}.npz") if isinstance(a, str) else a for a in cfg["arms"]
        ]
        arm_oof, arm_te, arm_info = [], [], []
        for p in paths:
            oof, te, ys, files = load_arm(p)
            if ys is not None and not np.array_equal(ys, Y):
                arm_info.append({"arm": p.stem, "Y_MISMATCH": True})
            arm_info.append(
                {
                    "arm": p.stem,
                    "oof_auc": round(float(roc_auc_score(Y, oof)), 5),
                    "nested": round(nested_auc(rank01(oof), Y), 5),
                    "y_matches_train": None if ys is None else bool(np.array_equal(ys, Y)),
                    "keys": files,
                }
            )
            arm_oof.append(rank01(oof))
            arm_te.append(rank01(te))

        f_oof = np.maximum.reduce(arm_oof)
        f_te = np.maximum.reduce(arm_te)
        labels = np.clip(f_te, 0.001, 0.999) if cfg["clip"] else f_te

        committed = pd.read_csv(cfg["csv"])["label"].values
        absdiff = np.abs(committed - labels)
        rep = {
            "arms": arm_info,
            "nested_5block": round(nested_auc(f_oof, Y), 5),
            "full_oof": round(float(roc_auc_score(Y, f_oof)), 5),
            "nested_10block": round(nested_auc(f_oof, Y, 10), 5),
            "repro_frac_diff": float(np.mean(absdiff > 1e-12)),
            "repro_max_abs": float(absdiff.max()),
            "repro_spearman": float(spearmanr(committed, labels).correlation),
            "csv_rows": int(len(committed)),
            "csv_label_range": [float(committed.min()), float(committed.max())],
        }
        results[name] = rep
        fusion_te[name] = committed  # use the COMMITTED vector for cross-compare
        fusion_oof[name] = f_oof
        print(f"== {name}")
        print(json.dumps(rep, indent=2, ensure_ascii=False))

    # cross-comparison against LB anchor (committed repo max3)
    anchor = pd.read_csv(V4 / "submissions/submission_v4_max3.csv")["label"].values
    ra = rankdata(anchor)
    n = len(anchor)
    print("\n== vs ANCHOR submission_v4_max3 (LB 0.71222) ==")
    cross = {}
    for name, v in fusion_te.items():
        rv = rankdata(v)
        sp = float(spearmanr(anchor, v).correlation)
        pear = float(np.corrcoef(ra, rv)[0, 1])
        moved = float(np.mean(np.abs(ra - rv) > 0.5))
        med_shift = float(np.median(np.abs(ra - rv)))
        p99_shift = float(np.percentile(np.abs(ra - rv), 99))
        # top-K overlap
        ov = {}
        for k in (100, 320, 640):
            a_top = set(np.argsort(-anchor)[:k])
            v_top = set(np.argsort(-v)[:k])
            ov[f"top{k}"] = round(len(a_top & v_top) / k, 4)
        identical = bool(np.array_equal(anchor, v))
        cross[name] = dict(
            spearman=round(sp, 6),
            rank_pearson=round(pear, 6),
            frac_rank_moved=round(moved, 4),
            median_rank_shift=med_shift,
            p99_rank_shift=p99_shift,
            identical=identical,
            **ov,
        )
        print(f"{name:18s} {json.dumps(cross[name])}")

    # OOF-level: same-ruler nested delta vs max3 fusion oof
    base_oof = fusion_oof["repo/v4_max3"]
    base_nested = nested_auc(base_oof, Y)
    print(f"\n== nested delta vs repo/v4_max3 (base nested={base_nested:.5f}) ==")
    for name, o in fusion_oof.items():
        print(f"{name:18s} nested={nested_auc(o, Y):.5f}  delta={nested_auc(o, Y)-base_nested:+.5f}")

    Path("/tmp/audit/out").mkdir(parents=True, exist_ok=True)
    Path("/tmp/audit/out/core.json").write_text(
        json.dumps({"per_candidate": results, "vs_anchor": cross}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
