import sys, time, json
sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import load_raw, add_parsed, make_splits, RAW_NOISE_COLS

tr,te=load_raw(); y=tr["label"].to_numpy(); trf=add_parsed(tr)
CAT_ALL=["month","region","code","source","grades","version","t3","t3s"]
sets={
 "full":[c for c in trf.columns if c not in ("id","label")],
 "clean":[c for c in trf.columns if c not in ("id","label") and c not in RAW_NOISE_COLS],
}
splits=make_splits(y,5,2)
for name,feats in sets.items():
    cats=[c for c in CAT_ALL if c in feats]
    X=trf[feats].copy()
    for c in cats: X[c]=X[c].astype(str)
    X=X.fillna({c:-999 for c in X.columns if c not in cats})
    for nit in [400,900]:
        oof={r:np.zeros(len(y)) for r in range(2)}
        t0=time.time()
        for r,f,ti,vi in splits:
            m=CatBoostClassifier(iterations=nit,depth=6,learning_rate=0.035,l2_leaf_reg=10,
                                 loss_function="Logloss",random_seed=2026+r,allow_writing_files=False,
                                 verbose=False,thread_count=4)
            m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
            oof[r][vi]=m.predict_proba(X.iloc[vi])[:,1]
        aucs=[roc_auc_score(y,oof[r]) for r in range(2)]
        print(name,nit,round(float(np.mean(aucs)),5),[round(a,5) for a in aucs],round(time.time()-t0,1),flush=True)
