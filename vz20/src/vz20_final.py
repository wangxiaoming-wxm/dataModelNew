"""vz20: 锁定配方后, 在全 train 上重拟合各臂, 预测 test, 生成 submission_vz20.csv.

full-train 重拟合 (非折内): 各臂 label-free 统计在全 train 上拟合, 训练 K seeds,
预测 test; byte07 用全 train 映射到 test. 应用 PROTOCOL 的 vz20 主配方.
"""
from __future__ import annotations
import os, sys, json, hashlib, argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vz20_arms as A  # noqa: E402

VZ20 = os.path.dirname(HERE)


def r(x):
    x = np.asarray(x, dtype=float)
    return rankdata(x) / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nseed", type=int, default=4)
    p.add_argument("--trees", type=int, default=800)
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--out", default=os.path.join(VZ20, "submission_vz20.csv"))
    args = p.parse_args()

    tr = pd.read_csv(os.path.join(VZ20, "..", "data", "train.csv"), dtype={"id": str})
    te = pd.read_csv(os.path.join(VZ20, "..", "data", "test.csv"), dtype={"id": str})
    y = tr["label"].astype(int).values
    tr_raw = tr.drop(columns=["label"]).reset_index(drop=True)

    preds = {}
    for arm in ["A1", "A2", "REF1", "REF2", "R1", "R2", "R3", "R4"]:
        cfg = A.ARMS[arm]
        seeds = (A.REF_BAG_SEEDS if cfg["ref"] else A.BAG_SEEDS)[:args.nseed]
        Xf, Xt, cats = A.build_world(cfg["world"], tr_raw, te)
        _, tp = A.train_arm_predict(cfg["world"], cfg, Xf, cats, y, Xf.iloc[:1], Xt,
                                    seeds, args.trees, args.threads)
        preds[arm] = tp
        print(f"  {arm} test pred ready ({len(seeds)}seed)", flush=True)

    # byte07 全 train -> test
    b0 = np.array([A.id_byte(x, 0) for x in tr["id"]])
    b7 = np.array([A.id_byte(x, 7) for x in tr["id"]])
    b0k = np.array([A.id_byte(x, 0) for x in te["id"]])
    b7k = np.array([A.id_byte(x, 7) for x in te["id"]])
    b0t = A.byte_te_map(b0, y, b0k)
    b7t = A.byte_te_map(b7, y, b7k)
    preds["BYTE07"] = (rankdata(b0t) + rankdata(b7t)) / 2 / len(te)

    # 锁定 vz20 主配方
    ref = r(0.5 * r(preds["REF1"]) + 0.5 * r(preds["REF2"]))
    ratio_family = np.mean([r(preds["A1"]), r(preds["R1"]), r(preds["R3"])], axis=0)
    rate_family = np.mean([r(preds["A2"]), r(preds["R2"]), r(preds["R4"])], axis=0)
    my_cb_v20 = r(0.64 * r(ratio_family) + 0.36 * r(rate_family))
    final = 0.89 * np.maximum(my_cb_v20, ref) + 0.11 * r(preds["BYTE07"])
    final = rankdata(final) / len(final)

    sub = pd.DataFrame({"id": te["id"], "label": np.clip(final, 0.001, 0.999)})
    sub.to_csv(args.out, index=False)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"saved {args.out} ({len(sub)} rows) sha256={sha}")
    with open(os.path.join(VZ20, "artifacts", "vz20", "final_sha.json"), "w") as f:
        json.dump({"submission": os.path.basename(args.out), "sha256": sha, "rows": len(sub)}, f, indent=2)


if __name__ == "__main__":
    main()
