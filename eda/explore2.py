import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',100)
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
al=pd.concat([tr.drop(columns=['label']),te],ignore_index=True)
print("=== x19 values ===")
print(sorted(al.x19.unique()))
print("x19 counts:\n", al.x19.value_counts().sort_index())
print("\n=== livability values ==="); print(sorted(al.livability.unique()))
print("\n=== source ==="); print(al.source.value_counts())
print("\n=== version ==="); print(al.version.value_counts())
print("\n=== region ==="); print(al.region.value_counts())
print("\n=== month ==="); print(al.month.value_counts())
print("\n=== age_range ==="); print(al.age_range.value_counts().sort_index())
print("\n=== x20 sample sorted ==="); print(sorted(al.x20.unique())[:40])
print("\n=== t3 sample ==="); print(al.t3.value_counts().head(20))
print("\n=== corr of x among themselves (abs>0.3) ===")
X=al[[f'x{i}' for i in range(21)]]
c=X.corr()
for i in range(21):
    for j in range(i+1,21):
        if abs(c.iloc[i,j])>0.25: print(f'x{i}-x{j}: {c.iloc[i,j]:.3f}')
print("\n=== corr x vs other numerics ===")
num=['days','cc','condition','V','max_g','age_range','livability','x19','x20']
cc2=al[[f'x{i}' for i in range(21)]+num].corr()
print(cc2.loc[[f'x{i}' for i in range(21)],num].round(3).to_string())
