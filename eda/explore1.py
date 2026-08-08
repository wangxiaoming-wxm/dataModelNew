import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 100)
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
print(tr.shape, te.shape)
print("label mean", tr.label.mean(), tr.label.sum())
print("\n=== dtypes / nunique / nan ===")
rows=[]
for c in tr.columns:
    s=tr[c]
    rows.append(dict(col=c, dtype=str(s.dtype), nuniq=s.nunique(dropna=False), nan=s.isna().sum(),
                     nuniq_te=(te[c].nunique(dropna=False) if c in te.columns else -1),
                     ex=str(list(s.dropna().unique()[:4]))[:70]))
print(pd.DataFrame(rows).to_string())
print("\n=== describe numeric ===")
print(tr.describe(include=[np.number]).T.to_string())
