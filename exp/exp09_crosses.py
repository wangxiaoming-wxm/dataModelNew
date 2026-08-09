"""How many categorical crosses does the CatBoost arm want?"""
import sys, time, json, itertools
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
CKPT=[300,500,700,900,1200]
P=dict(loss_function="Logloss",learning_rate=0.03,depth=6,l2_leaf_reg=10,
       random_strength=0.7,verbose=False,thread_count=4,allow_writing_files=False)
cfgs=[(0,0),(6,0),(10,0),(13,4),(16,8)]
for npb,ntr_ in cfgs:
    X,cats=build(tr,edges,"cross",n_pair_base=npb,n_triples=ntr_)
    for c in cats: X[c]=X[c].astype(str)
    X=X.fillna({c:-999.0 for c in X.columns if c not in cats})
    oof={k:np.zeros((NR,len(y))) for k in CKPT}
    t0=time.time()
    for r,f,ti,vi in splits:
        m=CatBoostClassifier(**P,iterations=max(CKPT),random_seed=100+r*11+f)
        m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        for k in CKPT:
            oof[k][r,vi]=m.predict_proba(X.iloc[vi],ntree_end=k)[:,1]
    line={k:(round(float(np.mean([roc_auc_score(y,oof[k][r]) for r in range(NR)])),5),
             round(float(roc_auc_score(y,np.mean([rankdata(oof[k][r]) for r in range(NR)],axis=0))),5)) for k in CKPT}
    print(f"pairs={npb} triples={ntr_} nfeat={X.shape[1]} ncat={len(cats)} {line} {time.time()-t0:.0f}s",flush=True)
