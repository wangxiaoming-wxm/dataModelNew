"""Quantify the early-stopping-on-outer-fold optimism used by the B7 pipeline."""
import sys, time
sys.path.insert(0,"src2")
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import load_raw, add_parsed, make_splits, RAW_NOISE_COLS

tr,te=load_raw(); y=tr["label"].to_numpy(); trf=add_parsed(tr)
CAT_ALL=["month","region","code","source","grades","version","t3","t3s"]
feats=[c for c in trf.columns if c not in ("id","label")]          # full raw, like B5/B6
cats=[c for c in CAT_ALL if c in feats]
X=trf[feats].copy()
for c in cats: X[c]=X[c].astype(str)
for c in X.columns:
    if c not in cats: X[c]=X[c].fillna(X[c].median())
splits=make_splits(y,5,2)

P=dict(loss_function="Logloss",eval_metric="AUC",learning_rate=0.03,depth=6,l2_leaf_reg=10,
       random_strength=0.7,verbose=False,thread_count=4,allow_writing_files=False)

for mode in ["leaky_es_on_valid","honest_fixed_400","honest_inner_es"]:
    oof={r:np.zeros(len(y)) for r in range(2)}; iters=[]
    t0=time.time()
    for r,f,ti,vi in splits:
        if mode=="leaky_es_on_valid":
            m=CatBoostClassifier(iterations=1400,od_type="Iter",od_wait=150,random_seed=2026+f,**P)
            m.fit(X.iloc[ti],y[ti],cat_features=cats,eval_set=(X.iloc[vi],y[vi]),use_best_model=True,verbose=False)
            iters.append(m.get_best_iteration())
        elif mode=="honest_fixed_400":
            m=CatBoostClassifier(iterations=400,random_seed=2026+f,**P)
            m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        else:
            rng=np.random.default_rng(2026+f); idx=ti.copy(); rng.shuffle(idx)
            k=int(0.15*len(idx)); ei,ii=idx[:k],idx[k:]
            m0=CatBoostClassifier(iterations=1400,od_type="Iter",od_wait=150,random_seed=2026+f,**P)
            m0.fit(X.iloc[ii],y[ii],cat_features=cats,eval_set=(X.iloc[ei],y[ei]),use_best_model=True,verbose=False)
            b=max(1,m0.get_best_iteration()+1); iters.append(b)
            m=CatBoostClassifier(iterations=b,random_seed=2026+f,**P)
            m.fit(X.iloc[ti],y[ti],cat_features=cats,verbose=False)
        oof[r][vi]=m.predict_proba(X.iloc[vi])[:,1]
    aucs=[roc_auc_score(y,oof[r]) for r in range(2)]
    print(mode, round(float(np.mean(aucs)),5), [round(a,5) for a in aucs],
          "iters:", (int(np.mean(iters)) if iters else 400), f"{time.time()-t0:.0f}s", flush=True)
