"""Score the B6 `gap` feature view under an honest protocol (no early stop on the outer fold)."""
import sys, time
sys.path.insert(0,"src"); sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from insurance_claim.train_b6 import build_gap
from common import make_splits

tr=pd.read_csv("data/train.csv"); te=pd.read_csv("data/test.csv")
y=tr["label"].to_numpy(); feats=tr.drop(columns=["label"])
NR=2; splits=make_splits(y,5,NR)
CKPT=[300,500,700,900]
P=dict(loss_function="Logloss",learning_rate=0.03,depth=6,l2_leaf_reg=10,random_strength=0.7,
       verbose=False,thread_count=4,allow_writing_files=False)
oof={k:np.zeros((NR,len(y))) for k in CKPT}
oof_es=np.zeros((NR,len(y)))
t0=time.time()
for r,f,ti,vi in splits:
    Xtr,Xva,_,cats=build_gap(feats.iloc[ti].reset_index(drop=True),
                             feats.iloc[vi].reset_index(drop=True), te.copy())
    m=CatBoostClassifier(**P,iterations=max(CKPT),random_seed=2026+f)
    m.fit(Xtr,y[ti],cat_features=cats,verbose=False)
    for k in CKPT: oof[k][r,vi]=m.predict_proba(Xva,ntree_end=k)[:,1]
    m2=CatBoostClassifier(**P,iterations=1400,od_type="Iter",od_wait=150,random_seed=2026+f)
    m2.fit(Xtr,y[ti],cat_features=cats,eval_set=(Xva,y[vi]),use_best_model=True,verbose=False)
    oof_es[r,vi]=m2.predict_proba(Xva)[:,1]
    print(f"r{r} f{f} done {time.time()-t0:.0f}s",flush=True)
for k in CKPT:
    print("honest",k,round(float(np.mean([roc_auc_score(y,oof[k][r]) for r in range(NR)])),5),
          "bag",round(roc_auc_score(y,np.mean([rankdata(oof[k][r]) for r in range(NR)],axis=0)),5))
print("ES-on-valid (B6 protocol)",round(float(np.mean([roc_auc_score(y,oof_es[r]) for r in range(NR)])),5),
      "bag",round(roc_auc_score(y,np.mean([rankdata(oof_es[r]) for r in range(NR)],axis=0)),5))
