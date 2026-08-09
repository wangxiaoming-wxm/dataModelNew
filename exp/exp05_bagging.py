"""Does averaging OOF across repeated CV partitions explain the 0.66 -> 0.70 gap?"""
import sys, time
sys.path.insert(0,"src2")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from common import load_raw, add_parsed, make_splits, RAW_NOISE_COLS
import lightgbm as lgb

tr,te=load_raw(); y=tr["label"].to_numpy(); trf=add_parsed(tr)
CAT_ALL=["month","region","code","source","grades","version","t3","t3s"]
feats=[c for c in trf.columns if c not in ("id","label") and c not in RAW_NOISE_COLS]
cats=[c for c in CAT_ALL if c in feats]
X=trf[feats].copy()
for c in cats: X[c]=X[c].astype("category")

NR=10
splits=make_splits(y,5,NR)
P=dict(objective="binary",learning_rate=0.03,num_leaves=16,min_child_samples=100,
       feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,
       n_estimators=120,verbose=-1,n_jobs=4)
oof=np.zeros((NR,len(y)))
t0=time.time()
for r,f,ti,vi in splits:
    m=lgb.LGBMClassifier(**P,random_state=1000+r*7+f)
    m.fit(X.iloc[ti],y[ti],categorical_feature=cats)
    oof[r,vi]=m.predict_proba(X.iloc[vi])[:,1]
print("per-repeat AUCs:",[round(roc_auc_score(y,oof[r]),5) for r in range(NR)])
for k in [1,2,3,4,6,8,10]:
    avg=np.mean([rankdata(oof[r]) for r in range(k)],axis=0)
    print(f"rank-avg of first {k} repeats -> AUC {roc_auc_score(y,avg):.5f}")
print(f"{time.time()-t0:.0f}s")
