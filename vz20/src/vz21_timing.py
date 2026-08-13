#!/usr/bin/env python3
"""Single-fold smoke test: per-model AUC and wall time, to plan the budget."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/src")
from vz21_models import ZOO  # noqa: E402
from vz21_pipeline import as_category, make_matrices  # noqa: E402

train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
y = train["label"].astype(int).to_numpy()

Xtr, Xte, cats = make_matrices(train, test)
print(f"features: {Xtr.shape[1]}  categorical: {len(cats)}")
_, levels = as_category(pd.concat([Xtr, Xte], ignore_index=True), cats)

tri, vai = next(iter(StratifiedKFold(5, shuffle=True, random_state=424242).split(Xtr, y)))
only = sys.argv[1:] or list(ZOO)
for name in only:
    t0 = time.time()
    try:
        va, te = ZOO[name](Xtr.iloc[tri], y[tri], Xtr.iloc[vai], Xte, cats, levels, 0)
        print(f"{name:<12} auc={roc_auc_score(y[vai], va):.5f}  {time.time()-t0:6.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{name:<12} FAILED {type(exc).__name__}: {exc}", flush=True)
