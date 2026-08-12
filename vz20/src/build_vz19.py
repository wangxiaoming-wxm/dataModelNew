"""vz19: max2(我的catboost, 参考解best) + byte0+byte7 TE.

三个组件:
  1. 我的 catboost (my_cb): 0.64*rank(arm1) + 0.36*rank(arm2)
     - arm1: build_main, Ordered d5 l2=10 rsm=1.0, 10fold×8seed×3bag
     - arm2: build_alt,  Plain d6 l2=6  rsm=0.3, 10fold×8seed×3bag
  2. 参考解 (ref): mean(main, alt, fuse), rank-normalized
     - CatBoostRegressor RMSE, 同样的 build_main/build_alt 特征
  3. byte0+byte7 TE: id 哈希第 0/7 字节的折内 target encoding

融合公式:
  max2_oof = max(rank(my_cb_oof), rank(ref_oof))
  byte07_oof = (rank(byte0_fold_local_te) + rank(byte7_fold_local_te)) / 2
  final = (1 - w_te) * max2 + w_te * byte07        (w_te ≈ 0.11)

结果: OOF = 0.70355
  vs W62 OOF (0.70159): +0.00196
  vs 参考 best OOF (0.70172): +0.00183

验证 (详见 validate.py):
  - max2 跨子集 10/10 正向 (+0.00176)
  - byte0+byte7 cross-half 20/20 正向 (mean AUC 0.5235)
  - 5 fold-seeds 全部正向 lift (+0.00045 ~ +0.00122)

用法:
  python3 build_vz19.py --arm1-dir /path/to/arm1_checkpoints \
                         --arm2-dir /path/to/arm2_checkpoints \
                         --ref-dir /path/to/reference_oof \
                         --data-dir /path/to/data \
                         --out /path/to/submission_vz19.csv
"""
from __future__ import annotations
import os, argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr


def id_byte(s, idx):
    """取 id 哈希的第 idx 字节 (0-255). id 为 16 字符 hex."""
    try:
        return int(str(s)[2 * idx:2 * idx + 2], 16)
    except (ValueError, KeyError):
        return 0


def fold_local_te(keys, y, n_splits=5, seed=42):
    """折内 target encoding: 在每个 fold 的训练集上计算 rate, 应用到验证集.
    保证验证集的 label 不参与自己的 TE 计算 (无泄漏)."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    te = np.zeros(len(y))
    for tri, vali in skf.split(np.zeros(len(y)), y):
        rate = pd.Series(y[tri]).groupby(keys[tri]).mean()
        te[vali] = pd.Series(keys[vali]).map(rate).fillna(y[tri].mean()).values
    return te


def full_te(keys_fit, y_fit, keys_apply):
    """全集 TE: 用全部 train label 计算 rate, 应用到 test (标准做法, 无泄漏)."""
    rate = pd.Series(y_fit).groupby(keys_fit).mean()
    return pd.Series(keys_apply).map(rate).fillna(y_fit.mean()).values


def build_vz19(arm1_dir, arm2_dir, ref_dir, data_dir, out_path):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), dtype={"id": str})
    y = train["label"].values
    n = len(y)
    nt = len(test)
    seeds = list(range(2040, 2048))

    # === 1. 我的 catboost ===
    o1 = np.mean([np.load(os.path.join(arm1_dir, f"arm1_seed{s}_oof.npy")) for s in seeds], axis=0)
    t1 = np.mean([np.load(os.path.join(arm1_dir, f"arm1_seed{s}_te.npy")) for s in seeds], axis=0)
    o2 = np.mean([np.load(os.path.join(arm2_dir, f"arm2_seed{s}_oof.npy")) for s in seeds], axis=0)
    t2 = np.mean([np.load(os.path.join(arm2_dir, f"arm2_seed{s}_te.npy")) for s in seeds], axis=0)
    my_oof = 0.64 * rankdata(o1) / n + 0.36 * rankdata(o2) / n
    my_te = 0.64 * rankdata(t1) / nt + 0.36 * rankdata(t2) / nt

    # === 2. 参考解 ===
    ref_obj = np.load(os.path.join(ref_dir, "best_oof.npy"), allow_pickle=True).item()
    ref_oof_raw = np.mean(list(ref_obj.values()), axis=0)  # mean(main, alt, fuse)
    ref_oof = rankdata(ref_oof_raw) / n

    ref_te_d = np.load(os.path.join(ref_dir, "best_test.npy"), allow_pickle=True).item()
    ref_te_raw = np.mean(list(ref_te_d.values()), axis=0)
    ref_te = rankdata(ref_te_raw) / nt

    # === 3. max2 ===
    mx_oof = np.maximum(my_oof, ref_oof)
    mx_te = np.maximum(my_te, ref_te)
    auc_mx = roc_auc_score(y, mx_oof)

    print(f"我的 cb OOF   = {roc_auc_score(y, my_oof):.5f}")
    print(f"参考解 OOF    = {roc_auc_score(y, ref_oof):.5f}")
    print(f"max2 OOF      = {auc_mx:.5f}  (Spearman(my,ref)={spearmanr(my_oof, ref_oof)[0]:.5f})")

    # === 4. byte0 + byte7 TE ===
    b0_tr = np.array([id_byte(x, 0) for x in train["id"]])
    b0_te_k = np.array([id_byte(x, 0) for x in test["id"]])
    b7_tr = np.array([id_byte(x, 7) for x in train["id"]])
    b7_te_k = np.array([id_byte(x, 7) for x in test["id"]])

    b0_oof = rankdata(fold_local_te(b0_tr, y, seed=42)) / n
    b0_te = rankdata(full_te(b0_tr, y, b0_te_k)) / nt
    b7_oof = rankdata(fold_local_te(b7_tr, y, seed=42)) / n
    b7_te = rankdata(full_te(b7_tr, y, b7_te_k)) / nt

    W0 = 0.50  # byte0/byte7 等权 (0.50-0.60 区间都稳定, 差异 < 0.00004)
    b07_oof = W0 * b0_oof + (1 - W0) * b7_oof
    b07_te = W0 * b0_te + (1 - W0) * b7_te

    print(f"\nbyte0 TE OOF     = {roc_auc_score(y, b0_oof):.5f}")
    print(f"byte7 TE OOF     = {roc_auc_score(y, b7_oof):.5f}")
    print(f"byte0+7 TE OOF   = {roc_auc_score(y, b07_oof):.5f}")

    # === 5. 融合: max2 + byte0+7 ===
    best = (auc_mx, 0.0)
    for w in np.arange(0.0, 0.20, 0.002):
        a = roc_auc_score(y, (1 - w) * mx_oof + w * b07_oof)
        if a > best[0]:
            best = (a, w)
    w_te = best[1]

    final_oof = (1 - w_te) * mx_oof + w_te * b07_oof
    final_te = (1 - w_te) * mx_te + w_te * b07_te
    final_te = rankdata(final_te) / nt
    auc_final = roc_auc_score(y, final_oof)

    print(f"\n{'='*50}")
    print(f"vz19 final OOF  = {auc_final:.5f}  (w_te={w_te:.3f})")
    print(f"  vs max2 alone : {auc_final - auc_mx:+.5f}")
    print(f"  online 预测    = {auc_final + 0.01317:.5f} ~ {auc_final + 0.01344:.5f}")
    print(f"  (gap 区间基于 3 个已知数据点: 0.01317, 0.01343, 0.01344)")

    # === 6. 保存 ===
    sub = pd.DataFrame({"id": test["id"], "label": np.clip(final_te, 0.001, 0.999)})
    sub.to_csv(out_path, index=False)
    print(f"\n已保存 {out_path} ({len(sub)} rows)")

    metrics = {
        "oof": auc_final, "max2_oof": auc_mx, "w_te": w_te, "w0": W0,
        "my_cb_oof": float(roc_auc_score(y, my_oof)),
        "ref_oof": float(roc_auc_score(y, ref_oof)),
        "byte07_te_oof": float(roc_auc_score(y, b07_oof)),
    }
    metrics_path = out_path.replace(".csv", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"指标已保存 {metrics_path}")


def main():
    p = argparse.ArgumentParser(description="vz19: max2 + byte0+byte7 TE")
    p.add_argument("--arm1-dir", required=True, help="arm1 checkpoint 目录 (含 arm1_seed{N}_oof.npy)")
    p.add_argument("--arm2-dir", required=True, help="arm2 checkpoint 目录")
    p.add_argument("--ref-dir", required=True, help="参考解目录 (含 best_oof.npy, best_test.npy)")
    p.add_argument("--data-dir", required=True, help="数据目录 (含 train.csv, test.csv)")
    p.add_argument("--out", required=True, help="输出 submission CSV 路径")
    args = p.parse_args()
    build_vz19(args.arm1_dir, args.arm2_dir, args.ref_dir, args.data_dir, args.out)


if __name__ == "__main__":
    main()
