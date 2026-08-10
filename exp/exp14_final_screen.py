import sys, time
sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import load_raw, make_splits
from features import fit_edges, build, add_noise_view, _derive
from jitter import add_jitter_views

tr,te=load_raw(); y=tr["label"].to_numpy()
allf=pd.concat([tr.drop(columns=["label"]),te],ignore_index=True)
edges=fit_edges(allf); der=_derive(tr,edges["__scale__"])
days=pd.to_numeric(tr["days"])
NR=2; splits=make_splits(y,5,NR); CKPT=[600,800,1000]

def make(nv, off):
    X,cats=build(tr,edges,"cross2"); add_noise_view(X,cats,tr)
    add_jitter_views(X,cats,tr,der["cond_r"],days,n_views=nv,stream_offset=off)
    for c in cats: X[c]=X[c].astype(str)
    return X.fillna({c:-999.0 for c in X.columns if c not in cats}), cats

CACHE={}
def get(nv,off):
    if (nv,off) not in CACHE: CACHE[(nv,off)]=make(nv,off)
    return CACHE[(nv,off)]

def run(tag, nv, per_model_stream, depth=6, l2=10):
    oof={k:np.zeros((NR,len(y))) for k in CKPT}; t0=time.time()
    for i,(r,f,ti,vi) in enumerate(splits):
        off = i+1 if per_model_stream else 0
        X,cats=get(nv,off)
        m=CatBoostClassifier(loss_function="Logloss",iterations=max(CKPT),learning_rate=0.03,depth=depth,
                             l2_leaf_reg=l2,random_strength=0.7,verbose=False,thread_count=4,
                             allow_writing_files=False,random_seed=100+r*11+f)
        m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        for k in CKPT: oof[k][r,vi]=m.predict_proba(X.iloc[vi],ntree_end=k)[:,1]
    line={k:(round(float(np.mean([roc_auc_score(y,oof[k][r]) for r in range(NR)])),5),
             round(float(roc_auc_score(y,np.mean([rankdata(oof[k][r]) for r in range(NR)],axis=0))),5)) for k in CKPT}
    print(f"{tag} {line} {time.time()-t0:.0f}s",flush=True)

run("J4 fixed streams      ",4,False)
run("J4 per-model streams  ",4,True)
run("J6 per-model streams  ",6,True)
run("J4 per-model depth7   ",4,True,depth=7)
run("J4 per-model depth5   ",4,True,depth=5)
