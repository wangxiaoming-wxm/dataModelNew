"""How large is the B6/B7 optimism when early stopping maximises validation AUC directly?

train_b5_focus.CAT_PARAMS sets eval_metric="AUC" and use_best_model=True with the
outer validation fold as eval_set, so the reported iteration is the one that
maximises the very metric the OOF is then scored with.
"""
import sys, time
sys.path.insert(0,"src"); sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from insurance_claim.train_b5_focus import CAT_PARAMS
from insurance_claim.train_b6 import build_gap

tr=pd.read_csv("data/train.csv"); y=tr["label"].to_numpy(); feats=tr.drop(columns=["label"])
oof_es=np.zeros(len(y)); oof_ll=np.zeros(len(y)); oof_fix=np.zeros(len(y)); best=[]
t0=time.time()
for f,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20290).split(feats,y)):
    Xtr,Xva,_,cats=build_gap(feats.iloc[ti].reset_index(drop=True),
                             feats.iloc[vi].reset_index(drop=True), tr.head(5).drop(columns=["label"]))
    p=dict(CAT_PARAMS); p["thread_count"]=4; p["random_seed"]=20290+f
    m=CatBoostClassifier(**p)                      # eval_metric="AUC", od_wait=150
    m.fit(Xtr,y[ti],cat_features=cats,eval_set=(Xva,y[vi]),use_best_model=True,verbose=False)
    oof_es[vi]=m.predict_proba(Xva)[:,1]; best.append(m.get_best_iteration())
    p2=dict(p); p2["eval_metric"]="Logloss"
    m2=CatBoostClassifier(**p2)
    m2.fit(Xtr,y[ti],cat_features=cats,eval_set=(Xva,y[vi]),use_best_model=True,verbose=False)
    oof_ll[vi]=m2.predict_proba(Xva)[:,1]
    p3=dict(p); p3.pop("od_type",None); p3.pop("od_wait",None); p3["iterations"]=500
    m3=CatBoostClassifier(**p3); m3.fit(Xtr,y[ti],cat_features=cats,verbose=False)
    oof_fix[vi]=m3.predict_proba(Xva)[:,1]
    print(f"fold {f} best_iter(AUC)={best[-1]} {time.time()-t0:.0f}s",flush=True)
print("early stop on valid AUC     :",round(roc_auc_score(y,oof_es),5))
print("early stop on valid Logloss :",round(roc_auc_score(y,oof_ll),5))
print("fixed 500 trees, no peeking :",round(roc_auc_score(y,oof_fix),5))
print("optimism from AUC early stop:",round(roc_auc_score(y,oof_es)-roc_auc_score(y,oof_fix),5))
