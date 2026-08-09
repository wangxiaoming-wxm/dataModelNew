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
NR=3; splits=make_splits(y,5,NR)
P=dict(loss_function="Logloss",iterations=700,learning_rate=0.04,depth=6,l2_leaf_reg=10,
       random_strength=0.7,verbose=False,thread_count=4,allow_writing_files=False)
for level in ["core","flat","cross"]:
    X,cats=build(tr,edges,level)
    for c in cats: X[c]=X[c].astype(str)
    X=X.fillna({c:-999.0 for c in X.columns if c not in cats})
    oof=np.zeros((NR,len(y))); t0=time.time()
    for r,f,ti,vi in splits:
        m=CatBoostClassifier(**P,random_seed=100+r*11+f)
        m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        oof[r,vi]=m.predict_proba(X.iloc[vi])[:,1]
    single=float(np.mean([roc_auc_score(y,oof[r]) for r in range(NR)]))
    bag=float(roc_auc_score(y,np.mean([rankdata(oof[r]) for r in range(NR)],axis=0)))
    print(level,"nfeat",X.shape[1],"ncat",len(cats),"single",round(single,5),"bag",round(bag,5),f"{time.time()-t0:.0f}s",flush=True)
