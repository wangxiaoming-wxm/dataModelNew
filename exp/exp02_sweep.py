import sys, time, json
sys.path.insert(0, "src2")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_raw, add_parsed, make_splits, RAW_NOISE_COLS
import lightgbm as lgb

tr, te = load_raw(); y = tr["label"].to_numpy(); trf = add_parsed(tr)
CAT_ALL = ["month", "region", "code", "source", "grades", "version", "t3", "t3s"]
feats = [c for c in trf.columns if c not in ("id","label") and c not in RAW_NOISE_COLS]
cats = [c for c in CAT_ALL if c in feats]
X = trf[feats].copy()
for c in cats: X[c] = X[c].astype("category")
splits = make_splits(y, 5, 2)

grid = [
  dict(num_leaves=4,  min_child_samples=100, learning_rate=0.03, n_estimators=1500, lambda_l2=10),
  dict(num_leaves=8,  min_child_samples=100, learning_rate=0.03, n_estimators=1000, lambda_l2=10),
  dict(num_leaves=16, min_child_samples=100, learning_rate=0.03, n_estimators=800,  lambda_l2=10),
  dict(num_leaves=31, min_child_samples=50,  learning_rate=0.03, n_estimators=600,  lambda_l2=10),
]
CKPT = [50,100,150,200,300,400,500,700,1000,1500]
for g in grid:
    n = g.pop("n_estimators")
    p = dict(objective="binary", feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             verbose=-1, n_jobs=4, n_estimators=n, **g)
    oof = {c: {r: np.zeros(len(y)) for r in range(2)} for c in CKPT if c <= n}
    for r, f, ti, vi in splits:
        m = lgb.LGBMClassifier(**p, random_state=1000+r)
        m.fit(X.iloc[ti], y[ti], categorical_feature=cats)
        for c in oof:
            oof[c][r][vi] = m.predict_proba(X.iloc[vi], num_iteration=c)[:,1]
    line = {c: round(float(np.mean([roc_auc_score(y, oof[c][r]) for r in range(2)])),5) for c in oof}
    print(g, line, flush=True)
