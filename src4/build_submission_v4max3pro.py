#!/usr/bin/env python3
"""Build the frozen V4max3pro submission from committed arm artifacts.

Recipe (frozen):
  max(
    rank(merger_ord8),
    rank(v2_cat_alt8),
    rank(ord_noxb_bag),
    rank(plus_strong),
    rank(noxb10),
  )
  then clip labels to [0.001, 0.999]

This is the single source of truth for `submissions/submission_v4max3pro.csv`.
Re-running this script must bit-match the committed CSV (within float noise).

Usage:
  python3 src4/build_submission_v4max3pro.py
  python3 src4/build_submission_v4max3pro.py --check   # verify committed CSV matches rebuild
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUB = ROOT / "submissions"
ART_BASE = ROOT / "artifacts" / "v4max3"
ART_PRO = ROOT / "artifacts" / "v4max3pro"
N_BLOCKS = 5

# Frozen final members — do not reorder casually; order is documentation only.
FINAL_ARMS = [
    ("merger_ord8", ART_BASE / "merger_ord8.npz", "honest"),
    ("v2_cat_alt8", ART_BASE / "v2_cat_alt8.npz", "honest"),
    ("ord_noxb_bag", ART_BASE / "ord_noxb_bag.npz", "es"),
    ("plus_strong", ART_PRO / "plus_strong.npz", "plus10"),
    ("noxb10", ART_PRO / "noxb10.npz", "plus10"),
]


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    if "test_pred" in d:
        te = np.asarray(d["test_pred"], dtype=float)
    elif "test" in d:
        te = np.asarray(d["test"], dtype=float)
    else:
        raise KeyError(f"no test key in {path}")
    return oof, te, d


def nested_auc(y: np.ndarray, oof: np.ndarray, n_blocks: int = N_BLOCKS) -> float:
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def build():
    y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
    test = pd.read_csv(DATA / "test.csv")
    base_sub = pd.read_csv(SUB / "submission_v4_max3.csv")["label"].values

    arms = {}
    for name, path, tag in FINAL_ARMS:
        if not path.exists():
            raise FileNotFoundError(path)
        oof, te, raw = load_npz(path)
        if oof.shape[0] != 14930 or te.shape[0] != 6398:
            raise ValueError(f"{name} bad shape oof={oof.shape} te={te.shape}")
        arms[name] = {
            "tag": tag,
            "oof_raw": oof,
            "te_raw": te,
            "oof": rankdata(oof) / len(oof),
            "te": rankdata(te) / len(te),
            "oof_auc": float(roc_auc_score(y, oof)),
            "meta": {
                k: (np.asarray(raw[k]).tolist() if hasattr(raw[k], "tolist") else str(raw[k]))
                for k in raw.files
                if k in ("per_seed", "seeds", "pool")
            },
        }

    max3_names = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
    max3_oof = np.maximum.reduce([arms[n]["oof"] for n in max3_names])
    max3_te = np.maximum.reduce([arms[n]["te"] for n in max3_names])
    max3_nested = nested_auc(y, max3_oof)

    final_names = [n for n, _, _ in FINAL_ARMS]
    final_oof = np.maximum.reduce([arms[n]["oof"] for n in final_names])
    final_te = np.maximum.reduce([arms[n]["te"] for n in final_names])
    final_nested = nested_auc(y, final_oof)
    final_full = float(roc_auc_score(y, final_oof))
    labels = np.clip(final_te, 0.001, 0.999)

    rng = np.random.default_rng(0)
    sanity = float(roc_auc_score(rng.permutation(y), final_oof))

    report = {
        "recipe": "max(" + ", ".join(final_names) + ")",
        "arms": {
            n: {
                "path": str(FINAL_ARMS[i][1].relative_to(ROOT)),
                "tag": arms[n]["tag"],
                "oof_auc": arms[n]["oof_auc"],
                "meta": arms[n]["meta"],
            }
            for i, (n, _, _) in enumerate(FINAL_ARMS)
        },
        "max3_nested": max3_nested,
        "max3_public_lb": 0.71222,
        "cv_to_lb_gap": 0.71222 - max3_nested,
        "final_nested": final_nested,
        "final_full_oof": final_full,
        "delta_vs_max3": final_nested - max3_nested,
        "expected_lb_if_same_gap": final_nested + (0.71222 - max3_nested),
        "spearman_vs_max3_submission": float(spearmanr(labels, base_sub).correlation),
        "spearman_vs_max3_rebuild": float(spearmanr(labels, max3_te).correlation),
        "shuffled_sanity_auc": sanity,
        "admit_for_7155_claim": False,
        "recommendation": (
            "DO NOT claim 0.7155; optional incremental only "
            "(~0.714 under optimistic max3 gap; plus/ES arms add PROTOCOL_RISK)"
        ),
    }
    return labels, test, report, max3_nested, final_nested


def write_outputs(labels, test, report):
    ART_PRO.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "submission_v4max3pro.csv"
    pd.DataFrame({"id": test["id"], "label": labels}).to_csv(out, index=False)
    (ART_PRO / "recipe_report.json").write_text(json.dumps(report, indent=2))
    (ART_PRO / "status_report.json").write_text(
        json.dumps(
            {
                "max3_nested": report["max3_nested"],
                "max3_lb": report["max3_public_lb"],
                "gap": report["cv_to_lb_gap"],
                "target_lb": 0.7155,
                "needed_nested": 0.7155 - report["cv_to_lb_gap"],
                "best": {
                    "name": "max3+plus_strong+noxb10",
                    "nested": report["final_nested"],
                    "delta": report["delta_vs_max3"],
                    "exp_lb": report["expected_lb_if_same_gap"],
                    "sp": report["spearman_vs_max3_submission"],
                },
                "noxb10_oof": report["arms"]["noxb10"]["oof_auc"],
                "admit_for_7155_claim": False,
                "recommendation": report["recommendation"],
            },
            indent=2,
        )
    )
    return out


def check_committed(labels):
    path = SUB / "submission_v4max3pro.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    committed = pd.read_csv(path)["label"].values
    absdiff = np.abs(committed - labels)
    frac = float(np.mean(absdiff > 1e-12))
    mx = float(np.max(absdiff))
    sp = float(spearmanr(committed, labels).correlation)
    ok = frac == 0.0 and mx < 1e-12
    print(f"check: frac_diff={frac:.6f} max_abs={mx:.3e} spearman={sp:.10f} ok={ok}")
    if not ok:
        raise SystemExit("COMMITTED CSV DOES NOT MATCH REBUILD")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify committed CSV only")
    ap.add_argument("--write", action="store_true", help="rewrite submission + reports")
    args = ap.parse_args()
    labels, test, report, max3_n, final_n = build()
    print(json.dumps({k: report[k] for k in report if k != "arms"}, indent=2))
    print("arms:")
    for n, info in report["arms"].items():
        print(f"  {n:14s} oof={info['oof_auc']:.5f} tag={info['tag']} path={info['path']}")
    if args.check and not args.write:
        check_committed(labels)
        return 0
    if args.write or not args.check:
        out = write_outputs(labels, test, report)
        print(f"wrote {out}")
        check_committed(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
