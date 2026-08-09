"""Screen candidate interaction crosses by honest nested-OOF target-encoding AUC."""
import sys, itertools, json
sys.path.insert(0,"src2")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import load_raw, make_splits
from features import fit_edges, build
from te import encode

tr,te=load_raw(); y=tr["label"].to_numpy()
allf=pd.concat([tr.drop(columns=["label"]),te],ignore_index=True)
edges=fit_edges(allf)
X,cats=build(tr,edges,"flat")
base=["region","source","days_q10","days_q20","cond_q10","age_cat","bin_pat","month","version"]
splits=[(ti,vi) for ti,vi in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)]

def oof_auc(series):
    o=np.zeros(len(y))
    for ti,vi in splits:
        _,(ev,)=encode(series.iloc[ti],y[ti],[series.iloc[vi]],smoothing=30.0,seed=1)
        o[vi]=ev
    return roc_auc_score(y,o), o

single={}
for c in base:
    a,o=oof_auc(X[c]); single[c]=(a,o); print(f"single {c:10s} {a:.4f}",flush=True)

print("\n=== pair crosses: AUC(cross) vs AUC(additive logit of parts) ===")
rows=[]
eps=1e-6
for a,b in itertools.combinations(base,2):
    s=X[a].astype(str)+"|"+X[b].astype(str)
    auc_c,_=oof_auc(s)
    la=np.log(np.clip(single[a][1],eps,1-eps)/(1-np.clip(single[a][1],eps,1-eps)))
    lb=np.log(np.clip(single[b][1],eps,1-eps)/(1-np.clip(single[b][1],eps,1-eps)))
    auc_add=roc_auc_score(y,la+lb)
    rows.append(dict(cross=f"{a}*{b}", ncat=s.nunique(), auc_cross=round(auc_c,4),
                     auc_add=round(auc_add,4), gain=round(auc_c-auc_add,4)))
df=pd.DataFrame(rows).sort_values("gain",ascending=False)
pd.set_option('display.width',200)
print(df.to_string())
