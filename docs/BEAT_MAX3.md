# Beat max3 — 冲超 0.71222

## 硬目标
公开榜 **> 0.71222**（`submission_v4_max3.csv`）。  
V4ext 实测 **0.71123**，证明「丢掉 ord_noxb_bag + 用诚实 nested 外推」失败。

## 协议
- 底座三臂冻结：`merger_ord8` + `v2_cat_alt8` + `ord_noxb_bag`
- 融合：`max(rank)`（与冠军相同，不是 mean）
- 允许混合 ES（与冠军同协议）
- **禁止** nested+0.0095 外推 LB
- 监督门禁：Δnested≥0.001、Spearman∈[0.985,0.997]、blocks+≥4/5、永不删 noxb

## 当前本地最优（相对 max3 nested 0.70307）
| 配方 | nested Δ | Spearman | 文件 |
|---|---:|---:|---|
| +plus+noxb10+w12+**noxb_new5** | **+0.00315** | 0.9918 | `submission_max3_stage_best.csv`（P1 进行中 interim） |
| +plus+noxb10+cat_w12_d5 | +0.00282 | 0.9918 | `submission_max3_best.csv` |
| +plus+cat_w12_d5 | +0.00238 | 0.9917 | `submission_max3_plus_w12.csv` |
| +plus+noxb10 | +0.00215 | 0.9917 | `submission_max3_pro.csv` |
| +plus_strong | +0.00146 | 0.9917 | `submission_max3_plus.csv` |

## 推荐交榜顺序
1. **`submissions/submission_max3_best.csv`**（最高本地门禁增益）
2. `submission_max3_pro.csv`（更保守的 5 臂）
3. `submission_max3_plus.csv`（最小改动）

实榜未回执前不宣称已超过 0.71222。

## 训练
```bash
bash run_beat_max3.sh            # P0 fuse + P1–P3 长训新 noxb 族
bash run_beat_max3_followup.sh   # plus_new8 + CoFEH/goldmine 方法论臂
```

## 方法论接入
见 `docs/METHODOLOGY_ADOPT.md`：`features_goldmine` + CoFEH 蒸馏算子 + Made-With-ML 残差互补 + 监督门禁。


> **策略对齐**：交榜以 `docs/DELIVERY_HQ.md` 为准；`stage_best`/`max3_best`/`pro` 因高相关堆臂风险勿交。
