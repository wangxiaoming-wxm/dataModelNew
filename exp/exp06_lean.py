"""Compare feature blocks with paired repeated CV (CatBoost + LightGBM)."""
import sys, time, json, itertools
sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import load_raw, add_parsed, make_splits

tr,te=load_raw(); y=tr["label"].to_numpy(); trf=add_parsed(tr)
REAL_NUM=["days","condition","age_range"]
BINS=["t1","t2","r1","r2","c1","c2","w1","w2"]
SETS={
 "core":      (REAL_NUM+BINS, ["region","source"]),
 "core+mvg":  (REAL_NUM+BINS, ["region","source","month","version","grades"]),
 "core+x18":  (REAL_NUM+BINS+["x18"], ["region","source"]),
 "core+t3x20":(REAL_NUM+BINS+["t3n","x20"], ["region","source","t3s"]),
 "core+all_noise": (REAL_NUM+BINS+["x18","t3n","x20","cc","max_g","V"]+[f"x{i}" for i in range(18)],
                    ["region","source","month","version","grades","code","t3","t3s","livability"]),
}
NR=4
splits=make_splits(y,5,NR)
P=dict(loss_function="Logloss",iterations=500,learning_rate=0.04,depth=6,l2_leaf_reg=10,
       random_strength=0.7,verbose=False,thread_count=4,allow_writing_files=False)
res={}
for name,(nums,cats) in SETS.items():
    feats=nums+cats
    X=trf[feats].copy()
    for c in cats: X[c]=X[c].astype(str)
    for c in nums: X[c]=pd.to_numeric(X[c]).fillna(X[c].median())
    oof=np.zeros((NR,len(y))); t0=time.time()
    for r,f,ti,vi in splits:
        m=CatBoostClassifier(**P,random_seed=100+r*11+f)
        m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        oof[r,vi]=m.predict_proba(X.iloc[vi])[:,1]
    single=float(np.mean([roc_auc_score(y,oof[r]) for r in range(NR)]))
    bag=float(roc_auc_score(y,np.mean([rankdata(oof[r]) for r in range(NR)],axis=0)))
    res[name]=dict(nfeat=len(feats),single=round(single,5),bag=round(bag,5),secs=round(time.time()-t0))
    print(name,res[name],flush=True)
    np.save(f"exp/oof_{name.replace('+','_')}.npy",oof)
print(json.dumps(res,indent=2))
