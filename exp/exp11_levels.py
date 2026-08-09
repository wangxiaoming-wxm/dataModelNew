"""Compare FE levels x CatBoost ctr complexity, evaluating several tree counts per fit."""
import sys, time, json
sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import load_raw, make_splits
from features import fit_edges, build

tr,te=load_raw(); y=tr["label"].to_numpy()
allf=pd.concat([tr.drop(columns=["label"]),te],ignore_index=True)
edges=fit_edges(allf)
NR=2; splits=make_splits(y,5,NR)
CKPT=[300,500,700,1000,1400]
cfgs=[("cross2",4)]
for level,mcc in cfgs:
    X,cats=build(tr,edges,level)
    for c in cats: X[c]=X[c].astype(str)
    X=X.fillna({c:-999.0 for c in X.columns if c not in cats})
    oof={k:np.zeros((NR,len(y))) for k in CKPT}; t0=time.time()
    for r,f,ti,vi in splits:
        m=CatBoostClassifier(loss_function="Logloss",iterations=max(CKPT),learning_rate=0.03,depth=6,
                             l2_leaf_reg=10,random_strength=0.7,max_ctr_complexity=mcc,
                             verbose=False,thread_count=4,allow_writing_files=False,random_seed=100+r*11+f)
        m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        for k in CKPT: oof[k][r,vi]=m.predict_proba(X.iloc[vi],ntree_end=k)[:,1]
    line={k:(round(float(np.mean([roc_auc_score(y,oof[k][r]) for r in range(NR)])),5),
             round(float(roc_auc_score(y,np.mean([rankdata(oof[k][r]) for r in range(NR)],axis=0))),5)) for k in CKPT}
    print(f"{level:8s} mcc={mcc} nfeat={X.shape[1]} ncat={len(cats)} {line} {time.time()-t0:.0f}s",flush=True)
