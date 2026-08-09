"""Independent reconnaissance of the raw files.

Nothing here reuses src2/.  The point is to look at the bytes on disk with
fresh eyes and check, from scratch, whether anything in this dataset could
support an AUC anywhere near 1.0.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN = "data/train.csv"
TEST = "data/test.csv"


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    tr = pd.read_csv(TRAIN, dtype=str)
    te = pd.read_csv(TEST, dtype=str)
    y = tr["label"].astype(int).to_numpy()
    print(f"train {tr.shape}  test {te.shape}")
    print(f"sha256 train {sha(TRAIN)}")
    print(f"sha256 test  {sha(TEST)}")
    print(f"positives {y.sum()} / {len(y)} = {y.mean():.6f}")

    feat = [c for c in tr.columns if c != "label"]
    print(f"\n--- per-column cardinality / dtype ---")
    for c in feat:
        u = tr[c].nunique(dropna=False)
        sample = tr[c].dropna().iloc[0] if tr[c].notna().any() else "NA"
        print(f"  {c:12s} nuniq={u:6d}  ex={str(sample)[:26]}")

    # ---- 1. exact duplicate feature vectors -----------------------------
    key_all = tr[feat].fillna("NA").astype(str).agg("|".join, axis=1)
    dup = key_all.duplicated().sum()
    print(f"\n[1] exact duplicate rows in train (all feature cols): {dup}")
    key_te = te[feat].fillna("NA").astype(str).agg("|".join, axis=1)
    overlap = len(set(key_all) & set(key_te))
    print(f"    train/test exact feature-vector overlap: {overlap}")

    # ---- 2. id structure -------------------------------------------------
    ids = tr["id"]
    print(f"\n[2] id: len set = {set(ids.str.len())}, "
          f"charset = {''.join(sorted(set(''.join(ids.head(4000)))))}")
    print(f"    unique ids train {ids.nunique()} / {len(ids)}; "
          f"train&test id overlap {len(set(ids) & set(te['id']))}")
    iv = ids.map(lambda s: int(s, 16)).to_numpy(dtype=object)
    ivf = np.array([float(v) for v in iv])
    print(f"    AUC(int(id,16))            = {roc_auc_score(y, ivf):.5f}")
    print(f"    AUC(row order)             = {roc_auc_score(y, np.arange(len(y))):.5f}")
    # bit-level: any single bit of the id predictive?
    bits = np.zeros((len(y), 64), dtype=np.int8)
    for i, v in enumerate(iv):
        for b in range(64):
            bits[i, b] = (v >> b) & 1
    bit_auc = np.array([roc_auc_score(y, bits[:, b]) for b in range(64)])
    print(f"    per-bit AUC: max |auc-.5| = {np.abs(bit_auc - .5).max():.5f} "
          f"at bit {int(np.abs(bit_auc - .5).argmax())}")
    # nibble-level chi2 against the label
    from scipy.stats import chi2_contingency
    worst = []
    for pos in range(16):
        nib = ids.str[pos]
        ct = pd.crosstab(nib, y)
        chi2, p, *_ = chi2_contingency(ct)
        worst.append((p, pos))
    worst.sort()
    print(f"    id nibble-vs-label chi2: smallest p = {worst[0][0]:.4f} at nibble {worst[0][1]} "
          f"(16 tests, Bonferroni threshold {0.05/16:.4f})")

    # ---- 3. decimal-precision artefacts ---------------------------------
    print("\n[3] decimal precision of numeric-looking columns")
    for c in feat:
        s = tr[c].dropna().astype(str)
        if not s.str.match(r"^-?\d+\.\d+$").all():
            continue
        dec = s.str.split(".").str[1]
        lens = Counter(dec.str.len())
        # trailing-digit distribution: a generator artefact would show up here
        last = Counter(dec.str.rstrip("0").str[-1:])
        print(f"  {c:10s} decimals={dict(lens)}  last-nonzero-digit={dict(sorted(last.items()))}")

    # ---- 4. label runs / ordering ---------------------------------------
    runs = 1 + int((y[1:] != y[:-1]).sum())
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    exp = 1 + 2 * n1 * n0 / len(y)
    sd = np.sqrt(2 * n1 * n0 * (2 * n1 * n0 - len(y)) / (len(y) ** 2 * (len(y) - 1)))
    print(f"\n[4] runs test on label order: runs={runs} expected={exp:.1f} sd={sd:.1f} "
          f"z={(runs - exp) / sd:+.2f}")
    # block-mean label rate: is the file sorted/blocked in any way?
    blk = pd.Series(y).groupby(np.arange(len(y)) // 500).mean()
    print(f"    label rate over 500-row blocks: min={blk.min():.3f} max={blk.max():.3f} "
          f"sd={blk.std():.4f} (binomial sd={np.sqrt(.1*.9/500):.4f})")

    json.dump({"n_train": len(tr), "n_test": len(te), "pos_rate": float(y.mean()),
               "dup_rows": int(dup), "train_test_overlap": int(overlap),
               "max_bit_auc_dev": float(np.abs(bit_auc - .5).max())},
              open("hunt/out_h01.json", "w"), indent=2)


if __name__ == "__main__":
    main()
