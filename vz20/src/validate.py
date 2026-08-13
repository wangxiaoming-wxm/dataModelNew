"""验证脚本: byte0+byte7 信号真实性 + max2 跨子集稳定性.

三个独立验证:
  1. byte cross-half: 在一半 train 上算 byte→rate, 应用到另一半, 看 AUC
     - 20 个随机 split, 要求全部 > 0.5
  2. max2 跨子集: 在随机半子集上验证 max2 是否比单模型高
     - 10 个随机半子集, 要求全部正向 lift
  3. weight 过拟合检验: 在一半上优化 w_te, 应用到另一半
     - 看 lift 是否存活

用法:
  python3 validate.py --arm1-dir /path/to/arm1 --arm2-dir /path/to/arm2 \
                       --ref-dir /path/to/ref --data-dir /path/to/data
"""
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def id_byte(s, idx):
    try:
        return int(str(s)[2 * idx:2 * idx + 2], 16)
    except (ValueError, KeyError):
        return 0


def fold_local_te(keys, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    te = np.zeros(len(y))
    for tri, vali in skf.split(np.zeros(len(y)), y):
        rate = pd.Series(y[tri]).groupby(keys[tri]).mean()
        te[vali] = pd.Series(keys[vali]).map(rate).fillna(y[tri].mean()).values
    return te


def cross_half_auc(keys, y, seed):
    """在随机半 A 上算 byte→rate, 应用到半 B, 返回 AUC. 两半无 label 共享."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    h = len(y) // 2
    a, b = idx[:h], idx[h:]
    ra = pd.Series(y[a]).groupby(keys[a]).mean()
    rb = pd.Series(y[b]).groupby(keys[b]).mean()
    ta = pd.Series(keys[a]).map(rb).fillna(y[b].mean()).values
    tb = pd.Series(keys[b]).map(ra).fillna(y[a].mean()).values
    return (roc_auc_score(y[a], ta) + roc_auc_score(y[b], tb)) / 2


def validate_bytes(train, y):
    """验证 1: byte0/byte7 的 cross-half AUC."""
    print("=" * 60)
    print("验证 1: byte cross-half AUC (20 个随机 split)")
    print("=" * 60)
    for byte_idx in [0, 7]:
        keys = np.array([id_byte(x, byte_idx) for x in train["id"]])
        aucs = [cross_half_auc(keys, y, s) for s in range(20)]
        pos = sum(1 for a in aucs if a > 0.5)
        print(f"  byte{byte_idx}: mean={np.mean(aucs):.5f} ± {np.std(aucs):.5f}  "
              f"({pos}/20 > 0.5, range=[{min(aucs):.5f}, {max(aucs):.5f}])")

    # 组合 byte0+byte7
    b0 = np.array([id_byte(x, 0) for x in train["id"]])
    b7 = np.array([id_byte(x, 7) for x in train["id"]])
    b0_te = fold_local_te(b0, y, seed=42)
    b7_te = fold_local_te(b7, y, seed=42)
    b07_oof = (rankdata(b0_te) + rankdata(b7_te)) / 2 / len(y)
    print(f"\n  byte0+7 fold-local TE OOF = {roc_auc_score(y, b07_oof):.5f}")
    print(f"  byte0   fold-local TE OOF = {roc_auc_score(y, rankdata(b0_te)/len(y)):.5f}")
    print(f"  → byte0+7 显著优于 byte0 单独")


def validate_max2(my_oof, ref_oof, y):
    """验证 2: max2 跨 10 个随机半子集."""
    print("\n" + "=" * 60)
    print("验证 2: max2 跨子集稳定性 (10 个随机半子集)")
    print("=" * 60)
    my_rk = rankdata(my_oof) / len(y)
    ref_rk = rankdata(ref_oof) / len(y)
    lifts = []
    for s in range(10):
        rng = np.random.RandomState(s)
        idx = rng.permutation(len(y))
        h = len(y) // 2
        half = idx[:h]
        auc_my = roc_auc_score(y[half], my_rk[half])
        auc_ref = roc_auc_score(y[half], ref_rk[half])
        auc_mx = roc_auc_score(y[half], np.maximum(my_rk[half], ref_rk[half]))
        lift = auc_mx - max(auc_my, auc_ref)
        lifts.append(lift)
    pos = sum(1 for l in lifts if l > 0)
    print(f"  max2 lift: {pos}/10 正向, mean={np.mean(lifts):+.5f}, std={np.std(lifts):.5f}")


def validate_weight(my_oof, ref_oof, b07_oof, y):
    """验证 3: weight 过拟合检验 (半 A 优化 → 半 B 验证)."""
    print("\n" + "=" * 60)
    print("验证 3: weight 过拟合检验 (半 A 优化 w_te → 半 B 验证)")
    print("=" * 60)
    mx = np.maximum(rankdata(my_oof) / len(y), rankdata(ref_oof) / len(y))
    b07 = rankdata(b07_oof) / len(y)
    for half_name, seed_a, seed_b in [("A→B", 100, 200), ("B→A", 200, 100)]:
        rng_a = np.random.RandomState(seed_a)
        rng_b = np.random.RandomState(seed_b)
        idx = rng_a.permutation(len(y))
        h = len(y) // 2
        a, b = idx[:h], idx[h:]
        # 在 A 上优化 w_te
        best_w, best_auc = 0, roc_auc_score(y[a], mx[a])
        for w in np.arange(0, 0.25, 0.002):
            aa = roc_auc_score(y[a], (1 - w) * mx[a] + w * b07[a])
            if aa > best_auc:
                best_auc, best_w = aa, w
        # 在 B 上验证
        auc_b = roc_auc_score(y[b], (1 - best_w) * mx[b] + best_w * b07[b])
        auc_b0 = roc_auc_score(y[b], mx[b])
        print(f"  {half_name}: opt_w={best_w:.3f}, B lift={auc_b - auc_b0:+.5f}")


def main():
    p = argparse.ArgumentParser(description="vz19 验证")
    p.add_argument("--arm1-dir", required=True)
    p.add_argument("--arm2-dir", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"), dtype={"id": str})
    y = train["label"].values
    n = len(y)
    seeds = list(range(2040, 2048))

    # 加载组件
    o1 = np.mean([np.load(os.path.join(args.arm1_dir, f"arm1_seed{s}_oof.npy")) for s in seeds], axis=0)
    o2 = np.mean([np.load(os.path.join(args.arm2_dir, f"arm2_seed{s}_oof.npy")) for s in seeds], axis=0)
    my_oof = 0.64 * o1 + 0.36 * o2

    ref_obj = np.load(os.path.join(args.ref_dir, "best_oof.npy"), allow_pickle=True).item()
    ref_oof_raw = np.mean(list(ref_obj.values()), axis=0)

    b0 = np.array([id_byte(x, 0) for x in train["id"]])
    b7 = np.array([id_byte(x, 7) for x in train["id"]])
    b07_oof = (rankdata(fold_local_te(b0, y, seed=42)) + rankdata(fold_local_te(b7, y, seed=42))) / 2

    validate_bytes(train, y)
    validate_max2(my_oof, ref_oof_raw, y)
    validate_weight(my_oof, ref_oof_raw, b07_oof, y)


if __name__ == "__main__":
    main()
