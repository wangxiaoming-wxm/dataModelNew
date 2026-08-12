#!/usr/bin/env python3
"""E1: reproduce fp_v8's headline numbers and test what its "nested" measures.

The claim under review is OOF 0.77016 / nested 0.77033. In fuse_fp_v8.py the
"nested" number is:

    skf = StratifiedKFold(5, shuffle=True, random_state=2026)
    nest = mean(auc(y[va], fuse_o[va]) for _, va in skf.split(...))

i.e. it slices an already-finished score vector into five pieces and averages
the AUC of the pieces. Nothing is refit. This script proves that this quantity
cannot detect overfitting, by feeding it a vector that is 100% leakage.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/audit")
from common import ART, id_bytes, load_data  # noqa: E402

CACHE = "/tmp/audit_fpv8/artifacts/first_principles"


def their_nested(y, score, seed=2026):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean([roc_auc_score(y[va], score[va]) for _, va in skf.split(np.zeros(len(y)), y)]))


def main():
    train, test, _ = load_data()
    y = train["label"].astype(int).to_numpy()
    out = {}

    v8 = np.load(f"{CACHE}/v8_fuse_oof.npy")
    v7 = np.load(f"{CACHE}/v7_fuse_oof.npy")
    bm = np.load(f"{CACHE}/bytepair_mean_oof.npy")

    out["n_train"] = int(len(y))
    out["n_test"] = int(len(test))
    out["prior"] = float(y.mean())

    # --- 1a. the claimed numbers reproduce exactly from the cached vectors ---
    out["repro"] = {
        "v8_oof_auc": float(roc_auc_score(y, v8)),
        "v8_claimed_oof": 0.7701614508323125,
        "v8_nested_recomputed": their_nested(y, v8),
        "v8_claimed_nested": 0.7703295069355633,
        "v7_oof_auc": float(roc_auc_score(y, v7)),
        "bytepair_mean_auc": float(roc_auc_score(y, bm)),
        "recipe_check_max_abs_diff": float(np.max(np.abs(v8 - (0.80 * v7 + 0.20 * bm)))),
    }

    # --- 1b. "nested" is just the full AUC re-averaged; it is not a refit ---
    # Any fixed vector has fold-mean ~= full AUC. Show it on several vectors,
    # including one built from the labels themselves (perfect leakage).
    rng = np.random.default_rng(0)
    probes = {
        "v8_fuse": v8,
        "v7_fuse": v7,
        "pure_noise": rng.standard_normal(len(y)),
        # 100% leakage: the label plus a little noise. An honest nested CV
        # would still show ~0.5 generalization for a memorizer, but this
        # estimator happily reports ~1.0.
        "label_plus_noise_LEAK": y + 0.01 * rng.standard_normal(len(y)),
        # a "model" that memorises train exactly and would score 0.5 on test
        "rank_of_label_LEAK": (y * 1000.0 + rng.standard_normal(len(y))),
    }
    tab = {}
    for name, vec in probes.items():
        full = float(roc_auc_score(y, vec))
        nests = {f"seed{s}": their_nested(y, vec, s) for s in (2026, 7, 12345, 90210)}
        tab[name] = {
            "full_auc": full,
            "their_nested_seed2026": nests["seed2026"],
            "their_nested_all_seeds": nests,
            "max_abs_gap_to_full": float(max(abs(v - full) for v in nests.values())),
        }
    out["nested_is_a_reslice"] = tab

    # --- 1c. id structure sanity ---
    tr_b = id_bytes(train["id"])
    te_b = id_bytes(test["id"])
    out["id_structure"] = {
        "train_id_len_unique": sorted(set(train["id"].astype(str).str.len().tolist())),
        "train_ids_unique": int(train["id"].nunique()),
        "test_ids_unique": int(test["id"].nunique()),
        "id_overlap_train_test": int(len(set(train["id"]) & set(test["id"]))),
        # If ids are a uniform 64-bit hash, every byte is ~uniform on 0..255
        # and every bit is ~Bernoulli(0.5).
        "byte_mean_per_pos": [float(tr_b[:, i].mean()) for i in range(8)],
        "byte_uniform_expected_mean": 127.5,
        "bit_mean_min": float(min(((tr_b[:, b] >> j) & 1).mean() for b in range(8) for j in range(8))),
        "bit_mean_max": float(max(((tr_b[:, b] >> j) & 1).mean() for b in range(8) for j in range(8))),
        "test_byte_mean_per_pos": [float(te_b[:, i].mean()) for i in range(8)],
    }

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "e1_repro_nested.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
