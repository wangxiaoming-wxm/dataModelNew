import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',100)
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
al=pd.concat([tr.drop(columns=['label']),te],ignore_index=True)
xs=[f'x{i}' for i in range(21)]
print("=== within-source std / overall std for each x + numerics ===")
rows=[]
for c in xs+['days','cc','condition','max_g','livability','age_range','V','x20']:
    o=al[c].std(); w=al.groupby('source')[c].std().mean()
    wr=al.groupby('region')[c].std().mean()
    wc=al.groupby(['source','region'])[c].std().mean()
    rows.append(dict(col=c, overall=round(o,5), within_source=round(w,5), r_src=round(w/o,3),
                     within_region=round(wr,5), r_reg=round(wr/o,3), within_both=round(wc,5), r_both=round(wc/o,3)))
print(pd.DataFrame(rows).to_string())

print("\n=== univariate AUC on train (numeric raw) ===")
y=tr.label.values
res=[]
for c in tr.columns:
    if c in ('id','label'): continue
    s=tr[c]
    if s.dtype==object:
        m=tr.groupby(c)['label'].mean(); v=tr[c].map(m).values
    else:
        v=s.fillna(s.median()).values
    try:
        a=roc_auc_score(y,v); res.append((c, round(a,4), round(max(a,1-a),4)))
    except Exception as e: pass
res.sort(key=lambda t:-t[2])
print(pd.DataFrame(res, columns=['col','auc','auc_dir']).to_string())
