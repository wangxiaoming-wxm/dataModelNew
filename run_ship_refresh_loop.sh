#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1
last=0
while true; do
  n=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
  if [[ "$n" -gt "$last" ]]; then
    echo "[refresh] parts=$n $(date -Is)"
    python3 src_beat/build_ship_candidates.py | tee -a logs/beat_max3/ship_refresh.log
    last=$n
  fi
  # if probes admitted, try fuse
  if [[ -f artifacts/beat_max3/probes/summary.json ]]; then
    python3 - <<'PY' || true
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
res=json.loads(Path('artifacts/beat_max3/probes/summary.json').read_text())
adm=[k for k,v in res.items() if v.get('admit_to_max')]
if not adm:
    raise SystemExit(0)
y=pd.read_csv('data/train.csv')['label'].astype(int).values
tid=pd.read_csv('data/test.csv')['id']

def rk(a): return rankdata(np.asarray(a,float))/len(a)
def load(n):
    d=np.load(f'artifacts/beat_max3/{n}.npz', allow_pickle=True)
    return np.asarray(d['oof'],float), np.asarray(d['test_pred'] if 'test_pred' in d.files else d['test'],float)
def nested(oof):
    out=np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)),5):
        out[b]=rankdata(oof[b])/len(b)
    return roc_auc_score(y,out)
mo,mot=load('merger_ord8'); ca,cat=load('v2_cat_alt8')
# prefer strong third if exists
if Path('artifacts/beat_max3/ord_noxb_strong.npz').exists():
    od,odt=load('ord_noxb_strong')
else:
    od,odt=load('ord_noxb_bag')
pl,plt=load('plus_strong')
base=np.maximum.reduce([rk(mo),rk(ca),rk(od)])
baset=np.maximum.reduce([rk(mot),rk(cat),rk(odt)])
bn=nested(base)
for a in adm:
    path=f'probe_{a}' if a!='exp3' else 'probe_exp3'
    if not Path(f'artifacts/beat_max3/{path}.npz').exists():
        continue
    o,t=load(path)
    fo=np.maximum.reduce([rk(mo),rk(ca),rk(od),rk(pl),rk(o)])
    ft=np.maximum.reduce([rk(mot),rk(cat),rk(odt),rk(plt),rk(t)])
    nest=nested(fo); sp=float(spearmanr(ft,baset).correlation)
    delta=nest-bn
    print('ortho',a,'delta',delta,'sp',sp)
    if delta>=0.001 and 0.985<=sp<=0.997:
        tag=f'ship_ortho_{a}'
        pd.DataFrame({'id':tid,'label':ft}).to_csv(f'submissions/submission_{tag}.csv', index=False)
        # if better than current beat_max3, promote
        cur=json.loads(Path('artifacts/beat_max3/report_beat_max3.json').read_text()) if Path('artifacts/beat_max3/report_beat_max3.json').exists() else {'delta':-1}
        if delta>cur.get('delta',-1):
            Path('submissions/submission_beat_max3.csv').write_bytes(Path(f'submissions/submission_{tag}.csv').read_bytes())
            Path('artifacts/beat_max3/report_beat_max3.json').write_text(json.dumps({'tag':tag,'delta':delta,'nested':nest,'sp':sp},indent=2))
            print('PROMOTED', tag)
PY
  fi
  [[ "$n" -ge 16 ]] && grep -q PROBES_NOW_DONE logs/beat_max3/probes_now.log 2>/dev/null && break
  sleep 90
done
echo SHIP_REFRESH_DONE
