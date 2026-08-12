#!/usr/bin/env python3
"""E5: dataset diagnostics the previous rounds never ran.

  * adversarial validation: is test drawn from the same distribution as train?
  * duplicate / near-duplicate structure between train and test
  * key-space cardinality and unseen-key rates for the id families fp_v8 uses
  * is `days` a disguised time index (would make the split temporal)?
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/audit")
from common import ART, id_bytes, load_data  # noqa: E402

NUM = ["days", "cc", "condition", "V", "max_g", "age_range", "livability", "x19", "x20"] + [f"x{i}" for i in range(19)]
CAT = ["month", "region", "code", "t3", "source", "grades", "version"]
BIN = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]


def adversarial(train, test):
    import lightgbm as lgb

    both = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    z = np.r_[np.zeros(len(train)), np.ones(len(test))]
    X = pd.DataFrame(index=both.index)
    for c in NUM + BIN:
        X[c] = pd.to_numeric(both[c], errors="coerce")
    for c in CAT:
        X[c] = both[c].astype(str).astype("category")
    oof = np.zeros(len(z))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
        m.fit(X.iloc[a], z[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    auc = float(roc_auc_score(z, oof))
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(X, z)
    imp = sorted(zip(X.columns, m.feature_importances_), key=lambda t: -t[1])[:12]
    return auc, [(c, int(v)) for c, v in imp]


def main():
    train, test, _ = load_data()
    y = train["label"].astype(int).to_numpy()
    out = {}

    # ---- adversarial validation ----
    auc, imp = adversarial(train, test)
    out["adversarial"] = {
        "auc": auc,
        "verdict": "same distribution (iid split)" if auc < 0.55 else "DISTRIBUTION SHIFT",
        "top_features": imp,
    }
    print(f"adversarial AUC = {auc:.4f}  -> {out['adversarial']['verdict']}")
    print("  top:", imp[:6])

    # ---- duplicate structure ----
    feat_cols = [c for c in train.columns if c not in ("id", "label")]

    def rowkey(df, cols):
        return pd.Series(["|".join(map(str, r)) for r in df[cols].to_numpy()], index=df.index)

    tr_key = rowkey(train, feat_cols)
    te_key = rowkey(test, feat_cols)
    out["duplicates"] = {
        "train_exact_dup_rows": int(len(tr_key) - tr_key.nunique()),
        "test_exact_dup_rows": int(len(te_key) - te_key.nunique()),
        "train_test_exact_matches": int(len(set(tr_key) & set(te_key))),
    }
    # coarse "same entity" probe: identical on the low-cardinality descriptive block
    ent = ["region", "source", "month", "version", "code", "grades", "age_range", "x19", "livability"] + BIN
    tk = rowkey(train, ent)
    ek = rowkey(test, ent)
    out["duplicates"]["entity_block_train_groups"] = int(tk.nunique())
    out["duplicates"]["entity_block_shared_with_test"] = int(len(set(tk) & set(ek)))
    print("duplicates:", out["duplicates"])

    # ---- is `days` a time index? ----
    d = train["days"].to_numpy(float)
    out["days_as_time"] = {
        "train_min": float(d.min()),
        "train_max": float(d.max()),
        "test_min": float(test["days"].min()),
        "test_max": float(test["days"].max()),
        "train_unique": int(train["days"].nunique()),
        # if the split were temporal, test days would occupy a distinct range
        "overlap_ratio": float(
            (np.sum((test["days"].to_numpy(float) >= d.min()) & (test["days"].to_numpy(float) <= d.max())) / len(test))
        ),
        "corr_days_vs_month": float(
            np.corrcoef(d, train["month"].astype(str).str[1:].astype(int).to_numpy(float))[0, 1]
        ),
    }
    print("days_as_time:", out["days_as_time"])

    # ---- id key-space cardinality / unseen rates (E5 of the brief) ----
    tb, teb = id_bytes(train["id"]), id_bytes(test["id"])
    card = {}
    def cov(name, trk, tek):
        u_tr = len(np.unique(trk))
        unseen = float(np.mean(~np.isin(tek, np.unique(trk))))
        card[name] = {
            "levels_in_train": u_tr,
            "levels_in_test": int(len(np.unique(tek))),
            "rows_per_level_train": round(len(trk) / max(u_tr, 1), 2),
            "test_rows_with_unseen_key": round(unseen, 4),
        }

    cov("byte", tb[:, 0], teb[:, 0])
    cov("byte_hi", tb[:, 0] >> 4, teb[:, 0] >> 4)
    cov("bit", (tb[:, 0] & 1), (teb[:, 0] & 1))
    cov("bytepair_xor", tb[:, 0] ^ tb[:, 1], teb[:, 0] ^ teb[:, 1])
    cov("bytepair_and", tb[:, 0] & tb[:, 1], teb[:, 0] & teb[:, 1])
    cov("bytepair_concat", tb[:, 0].astype(int) * 256 + tb[:, 1], teb[:, 0].astype(int) * 256 + teb[:, 1])
    out["id_cardinality"] = card
    print("id cardinality:", json.dumps(card, indent=2))

    # positives per level is what actually governs TE variance
    out["positives"] = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    out["id_cardinality"]["byte"]["positives_per_level"] = round(float(y.sum()) / 256, 2)
    out["id_cardinality"]["bytepair_concat"]["positives_per_level"] = round(float(y.sum()) / 65536, 4)

    (ART / "e4_diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", ART / "e4_diagnostics.json")


if __name__ == "__main__":
    main()
