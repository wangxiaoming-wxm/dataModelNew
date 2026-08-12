"""vz20 置乱哨兵: 打乱 y 后重训代表臂, held-out AUC 应 ∈ [0.48, 0.52].

若打乱标签仍得到明显 >0.52 的 AUC, 说明存在泄漏. 用 R1(ratio_rich) 作代表臂,
在 3 个外折上评估, 报告 fold-mean 与 pooled.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vz20_arms as A  # noqa: E402

VZ20 = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R1")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--nseed", type=int, default=2)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--perm-seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=-1)
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(VZ20, "..", "data", "train.csv"), dtype={"id": str})
    te = pd.read_csv(os.path.join(VZ20, "..", "data", "test.csv"), dtype={"id": str})
    y = tr["label"].astype(int).values
    tr_raw = tr.drop(columns=["label"]).reset_index(drop=True)

    rng = np.random.RandomState(args.perm_seed)
    y_perm = y.copy()
    rng.shuffle(y_perm)

    skf = StratifiedKFold(5, shuffle=True, random_state=A.OUTER_SEED)
    folds = list(skf.split(tr_raw, y_perm))[: args.folds]
    cfg = A.ARMS[args.arm]
    seeds = A.BAG_SEEDS[: args.nseed]

    aucs = []
    pooled = np.zeros(len(y))
    seen = np.zeros(len(y), dtype=int)
    for i, (tri, vali) in enumerate(folds):
        fit_frame = tr_raw.iloc[tri].reset_index(drop=True)
        valid_frame = tr_raw.iloc[vali].reset_index(drop=True)
        Xf, Xv, cats = A.build_world(cfg["world"], fit_frame, valid_frame)
        vp, _ = A.train_arm_predict(cfg["world"], cfg, Xf, cats, y_perm[tri], Xv,
                                    Xv.iloc[:1], seeds, args.trees, args.threads)
        a = roc_auc_score(y_perm[vali], vp)
        aucs.append(float(a))
        pooled[vali] = vp
        seen[vali] = 1
        print(f"  perm fold{i}: AUC={a:.5f}", flush=True)

    mask = seen == 1
    pooled_auc = float(roc_auc_score(y_perm[mask], pooled[mask]))
    res = {
        "arm": args.arm, "perm_seed": args.perm_seed, "folds": args.folds,
        "fold_auc": aucs, "fold_mean": float(np.mean(aucs)), "pooled_auc": pooled_auc,
        "gate_range": [0.48, 0.52],
        "gate_pass": bool(0.48 <= np.mean(aucs) <= 0.52 and 0.48 <= pooled_auc <= 0.52),
    }
    os.makedirs(os.path.join(VZ20, "artifacts", "vz20"), exist_ok=True)
    with open(os.path.join(VZ20, "artifacts", "vz20", "permutation.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
