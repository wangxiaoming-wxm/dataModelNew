"""vz20: 从缓存的 per-fold 臂预测装配预注册配方, 输出 held-out 对比与 metrics.

严格执行 docs/PROTOCOL.md 的预注册配方; 不在 outer-valid 上搜索任何权重.
"""
from __future__ import annotations
import os, json, argparse, hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

VZ20 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(VZ20, "artifacts", "vz20", "cache")


def r(x):
    x = np.asarray(x, dtype=float)
    return rankdata(x) / len(x)


def load(profile, arm, f, kind):
    return np.load(os.path.join(CACHE, f"{profile}_{arm}_fold{f}_{kind}.npy"))


# ---------- 预注册配方 (输入: 各臂 rank 预测 dict P) ----------
def rec_ref(P):
    return r(0.5 * r(P["REF1"]) + 0.5 * r(P["REF2"]))


def rec_vz19(P):
    my_cb = r(0.64 * r(P["A1"]) + 0.36 * r(P["A2"]))
    return 0.89 * np.maximum(my_cb, rec_ref(P)) + 0.11 * r(P["BYTE07"])


def rec_vz20(P):
    ratio_family = np.mean([r(P["A1"]), r(P["R1"]), r(P["R3"])], axis=0)
    rate_family = np.mean([r(P["A2"]), r(P["R2"]), r(P["R4"])], axis=0)
    my_cb_v20 = r(0.64 * r(ratio_family) + 0.36 * r(rate_family))
    return 0.89 * np.maximum(my_cb_v20, rec_ref(P)) + 0.11 * r(P["BYTE07"])


def rec_vz20_4arm(P):  # 诊断
    ratio_family = np.mean([r(P["A1"]), r(P["R1"])], axis=0)
    rate_family = np.mean([r(P["A2"]), r(P["R2"])], axis=0)
    my_cb_v20 = r(0.64 * r(ratio_family) + 0.36 * r(rate_family))
    return 0.89 * np.maximum(my_cb_v20, rec_ref(P)) + 0.11 * r(P["BYTE07"])


def rec_vz20_max3(P):  # 诊断
    ratio_family = np.mean([r(P["A1"]), r(P["R1"]), r(P["R3"])], axis=0)
    rate_family = np.mean([r(P["A2"]), r(P["R2"]), r(P["R4"])], axis=0)
    my_cb_v20 = r(0.64 * r(ratio_family) + 0.36 * r(rate_family))
    rich = r(0.65 * r(P["R1"]) + 0.35 * r(P["R2"]))
    return 0.89 * np.maximum.reduce([my_cb_v20, rec_ref(P), rich]) + 0.11 * r(P["BYTE07"])


RECIPES = {
    "vz19": rec_vz19,
    "vz20": rec_vz20,
    "vz20_4arm": rec_vz20_4arm,
    "vz20_max3": rec_vz20_max3,
}
ARMS_ALL = ["A1", "A2", "REF1", "REF2", "R1", "R2", "R3", "R4", "BYTE07"]


def evaluate(profile, outer_splits):
    d = np.load(os.path.join(CACHE, f"folds_{profile}.npz"))
    y = d["y"]
    per_fold = {name: [] for name in RECIPES}
    pooled_pred = {name: np.zeros(len(y)) for name in RECIPES}
    arm_fold_auc = {a: [] for a in ARMS_ALL}
    for f in range(outer_splits):
        vi = d[f"valid_{f}"]
        yv = y[vi]
        P = {a: load(profile, a, f, "valid") for a in ARMS_ALL}
        for a in ARMS_ALL:
            arm_fold_auc[a].append(float(roc_auc_score(yv, P[a])))
        for name, fn in RECIPES.items():
            p = fn(P)
            per_fold[name].append(float(roc_auc_score(yv, p)))
            pooled_pred[name][vi] = p
    out = {"profile": profile, "outer_splits": outer_splits, "outer_seed": 90210}
    out["arm_mean_auc"] = {a: float(np.mean(v)) for a, v in arm_fold_auc.items()}
    out["recipes"] = {}
    for name in RECIPES:
        fa = per_fold[name]
        out["recipes"][name] = {
            "fold_auc": fa,
            "fold_mean": float(np.mean(fa)),
            "fold_std": float(np.std(fa, ddof=1)),
            "pooled_auc": float(roc_auc_score(y, pooled_pred[name])),
        }
    # 配对 lift vs vz19
    v19 = per_fold["vz19"]
    out["lift_vs_vz19"] = {}
    for name in RECIPES:
        if name == "vz19":
            continue
        diffs = [per_fold[name][f] - v19[f] for f in range(outer_splits)]
        out["lift_vs_vz19"][name] = {
            "per_fold_diff": diffs,
            "fold_mean_lift": float(np.mean(diffs)),
            "folds_not_degraded": int(sum(1 for x in diffs if x >= -1e-6)),
            "n_folds": outer_splits,
        }
    return out, pooled_pred, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="full")
    p.add_argument("--outer-splits", type=int, default=5)
    args = p.parse_args()
    out, _, _ = evaluate(args.profile, args.outer_splits)
    print(json.dumps(out, indent=2))
    print("\n=== 门禁裁决 (主配方 vz20) ===")
    L = out["lift_vs_vz19"]["vz20"]
    g = out["recipes"]
    print(f"vz19 fold-mean = {g['vz19']['fold_mean']:.5f}  folds={['%.5f'%x for x in g['vz19']['fold_auc']]}")
    print(f"vz20 fold-mean = {g['vz20']['fold_mean']:.5f}  folds={['%.5f'%x for x in g['vz20']['fold_auc']]}")
    print(f"fold-mean lift = {L['fold_mean_lift']:+.5f}  (门禁 >=+0.0015)")
    print(f"folds not degraded = {L['folds_not_degraded']}/{L['n_folds']}  (门禁 >=3/5)")
    gate1 = L["fold_mean_lift"] >= 0.0015
    gate2 = L["folds_not_degraded"] >= 3
    print(f"GATE lift>=+0.0015: {'PASS' if gate1 else 'FAIL'}")
    print(f"GATE >=3/5 folds:   {'PASS' if gate2 else 'FAIL'}")
    print(f"诊断: vz20_4arm lift={out['lift_vs_vz19']['vz20_4arm']['fold_mean_lift']:+.5f}  "
          f"vz20_max3 lift={out['lift_vs_vz19']['vz20_max3']['fold_mean_lift']:+.5f}")


if __name__ == "__main__":
    main()
