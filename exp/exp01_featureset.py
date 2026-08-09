"""Experiment 01: does dropping the anonymisation-noise columns help?"""
import sys, time, json
sys.path.insert(0, "src2")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_raw, add_parsed, make_splits, RAW_NOISE_COLS
import lightgbm as lgb

tr, te = load_raw()
y = tr["label"].to_numpy()
trf, tef = add_parsed(tr), add_parsed(te)

CAT_ALL = ["month", "region", "code", "source", "grades", "version", "t3", "t3s"]

FEATSETS = {
    # everything the organisers gave us, lightly parsed
    "full": [c for c in trf.columns if c not in ("id", "label")],
    # drop the columns whose within-`source` residual is pure uniform noise
    "clean": [c for c in trf.columns if c not in ("id", "label") and c not in RAW_NOISE_COLS],
}
FEATSETS["clean_nolив"] = [c for c in FEATSETS["clean"] if c != "livability"]

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=16, min_child_samples=60,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, n_estimators=600, verbose=-1, n_jobs=4)

splits = make_splits(y, 5, 4)
out = {}
for name, feats in FEATSETS.items():
    cats = [c for c in CAT_ALL if c in feats]
    X = trf[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    oof = {r: np.zeros(len(y)) for r in range(4)}
    t0 = time.time()
    for r, f, ti, vi in splits:
        m = lgb.LGBMClassifier(**PARAMS, random_state=1000 + r)
        m.fit(X.iloc[ti], y[ti], categorical_feature=cats)
        oof[r][vi] = m.predict_proba(X.iloc[vi])[:, 1]
    aucs = [roc_auc_score(y, oof[r]) for r in range(4)]
    out[name] = dict(n_feat=len(feats), auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                     aucs=[round(a, 5) for a in aucs], secs=round(time.time() - t0, 1))
    print(name, json.dumps(out[name]), flush=True)
print(json.dumps(out, indent=2))
