import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',100)
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
al=pd.concat([tr.drop(columns=['label']),te],ignore_index=True)
xs=[f'x{i}' for i in range(18)]
# residual within source
R=al.groupby('source')[xs+['cc','max_g','V']].transform(lambda s: s-s.mean())
print("=== residual ranges (should be ~uniform width) ===")
print(R.describe().T[['std','min','max']].assign(width=lambda d:d['max']-d['min'], unif_std=lambda d:(d['max']-d['min'])/np.sqrt(12)).round(6).to_string())
print("\n=== residual correlation matrix (max offdiag) ===")
c=R[xs].corr().abs().values; np.fill_diagonal(c,0)
print("max abs offdiag corr among x residuals:", c.max())
print(pd.DataFrame(R[xs].corr().round(3)).to_string())
print("\nresid cc vs resid x corr:", R[['cc','max_g','V']+xs].corr().loc[['cc','max_g','V'],xs].abs().max(axis=1).to_dict())

print("\n=== univariate AUC ===")
y=tr.label.values; res=[]
for c_ in tr.columns:
    if c_ in ('id','label'): continue
    s=tr[c_]
    if s.dtype==object or str(s.dtype)=='string':
        m=tr.groupby(c_)['label'].mean(); v=tr[c_].map(m).astype(float).values
    else:
        v=pd.to_numeric(s,errors='coerce'); v=v.fillna(v.median()).values
    a=roc_auc_score(y,v); res.append((c_, round(a,4), round(max(a,1-a),4)))
res.sort(key=lambda t:-t[2])
print(pd.DataFrame(res, columns=['col','auc','auc_dir']).to_string())
