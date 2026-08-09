"""Two diagnostics that decide where the remaining headroom is.

1. Learning curve: shrink the fitting rows to 40/60/80/100 % while scoring on the
   same validation folds.  If AUC is still climbing at 100 %, we are limited by
   how much data each model sees, not by model capacity - and then 10-fold beats
   5-fold for free.
2. 5-fold vs 10-fold at full size.
"""
import sys, time, json
sys.path.insert(0, "src2")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from features import fit_edges
from arms import catboost_frame

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
y = tr["label"].to_numpy()
raw = pd.concat([tr.drop(columns=["label"]), te], ignore_index=True)
edges = fit_edges(raw)
X, cats = catboost_frame(raw, edges, stream_offset=1)
Xtr = X.iloc[:len(tr)].reset_index(drop=True)
P = dict(loss_function="Logloss", learning_rate=0.03, depth=5, l2_leaf_reg=10,
         random_strength=0.7, verbose=False, thread_count=4,
         allow_writing_files=False, iterations=1000)
NR = 2
out = {}

def run(tag, n_splits, frac):
    oof = np.zeros((NR, len(y))); t0 = time.time()
    for r in range(NR):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=31000 + r)
        for f, (ti, vi) in enumerate(skf.split(Xtr, y)):
            if frac < 1.0:
                rng = np.random.default_rng(900 + r * 17 + f)
                keep = rng.permutation(len(ti))[: int(round(frac * len(ti)))]
                ti = ti[np.sort(keep)]
            m = CatBoostClassifier(**P, random_seed=31000 + r * 11 + f)
            m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
            oof[r, vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
    single = float(np.mean([roc_auc_score(y, oof[r]) for r in range(NR)]))
    bag = float(roc_auc_score(y, np.mean([rankdata(oof[r]) for r in range(NR)], axis=0)))
    n_fit = int(round(frac * len(y) * (n_splits - 1) / n_splits))
    out[tag] = dict(n_fit_rows=n_fit, single=round(single, 5), bag=round(bag, 5),
                    secs=round(time.time() - t0))
    print(f"{tag:22s} fit_rows={n_fit:6d} single={single:.5f} bag={bag:.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)

run("5fold_40pct", 5, 0.40)
run("5fold_60pct", 5, 0.60)
run("5fold_80pct", 5, 0.80)
run("5fold_100pct", 5, 1.00)
run("10fold_100pct", 10, 1.00)
print(json.dumps(out, indent=2))
