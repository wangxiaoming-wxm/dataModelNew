"""Merge reference plus_v10 4 seeds with newly trained plus10 plain seeds into 8-seed bag."""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "v4max3pro"
y = None
oofs, tes, aucs, seeds = [], [], [], []

# original 4 seeds from reference file (has per-seed oof; need per-seed test?)
ref = np.load(ROOT / "artifacts" / "v4max3" / "plus_v10.npz")
# only aggregated test in plus_v10; use oof_seed_* and shared test as approx for old seeds
# Better: use reference file which has oof_seed but one test — for old seeds we only have bagged test.
# So for 8-seed we rank-pool: (mean of 4 old seed oofs already in ref['oof']) is 4-seed bag.
# Instead load new parts and average with old bag weighted 4:4

new_parts = sorted(ART.glob("part_plus10_plain_s*.npz"))
print("new parts", [p.name for p in new_parts])
if len(new_parts) < 4:
    raise SystemExit(f"need 4 new parts, have {len(new_parts)}")

old_oof = np.asarray(ref["oof"], float)
old_te = np.asarray(ref["test"], float)
y = np.asarray(ref["y"], int)

new_oofs = []
new_tes = []
for p in new_parts:
    d = np.load(p)
    new_oofs.append(d["oof"])
    new_tes.append(d["test"])
    aucs.append(float(roc_auc_score(y, d["oof"])))
    print(p.name, aucs[-1])

new_oof = np.mean(new_oofs, 0)
new_te = np.mean(new_tes, 0)
# equal weight old 4-seed bag and new 4-seed bag
oof = 0.5 * old_oof + 0.5 * new_oof
te = 0.5 * old_te + 0.5 * new_te
# also rank-pool version
oof_r = 0.5 * (rankdata(old_oof)/len(old_oof)) + 0.5 * (rankdata(new_oof)/len(new_oof))
te_r = 0.5 * (rankdata(old_te)/len(old_te)) + 0.5 * (rankdata(new_te)/len(new_te))
print("old bag", roc_auc_score(y, old_oof), "new bag", roc_auc_score(y, new_oof), "mix", roc_auc_score(y, oof), "rankmix", roc_auc_score(y, oof_r))
np.savez_compressed(ART / "plus_v10_8.npz", oof=oof_r, test_pred=te_r, y=y, pool="rankmix_4+4")
print("wrote plus_v10_8.npz")
