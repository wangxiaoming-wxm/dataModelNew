# SUPER714-Plus（差异化实验 — 已完成，未超越 W62）

相对已公开的 best_v1 / `submission_super714.csv`（线上 0.71453）刻意改配方。  
**结果：本地 max2 OOF 0.69645，相对 best_v1 的 −0.00483；不推荐线上提交。**

当前线上最强仍是 **W62 = 0.71503**（`task_w62`）。

## 与冠军的差异（实验配方）

| 维度 | best_v1 | SUPER714-Plus |
|---|---|---|
| main depth | Ordered **5** | Ordered **6** |
| alt l2 / 分箱 | l2=6，bins (7,13,25) | l2=**5**，bins **(6,12,24)** |
| 特征 | 单世界 | main **+rate**；alt **+ratio/cond_r** |
| seeds × bags × trees | 8 × 3 × 800 | **10 × 4 × 1000** |
| seed 起点 | 2026 | **3100** |

## 实测指标

| 融合 | OOF |
|---|---|
| Plus max2 | 0.69645 |
| Plus-w62 (0.62/0.38) | 0.69754 |
| Plus-wbest (w_main=0.80) | 0.69803 |
| best_v1 max2 | 0.70128 |
| **best_v1 W62** | **0.70159**（线上 **0.71503**） |

test Spearman(Plus-max2, best_v1-max2) ≈ 0.983（文件不同，但更弱）。

## 复现

```bash
bash run_super714_plus.sh          # 全量（很慢）
python3 -u src_super/fuse_plus_weights.py   # 生成 max2/w62/wbest
```

产物见 `artifacts/super714_plus/` 与 `submissions/submission_super714_plus*.csv`。

## 教训

加深 depth、改分箱、跨世界连续特征在本任务上**伤害**双世界信号；后续增益应在 best_v1 同构轴上找（bagging / 加权融合），而不是再改世界定义。
