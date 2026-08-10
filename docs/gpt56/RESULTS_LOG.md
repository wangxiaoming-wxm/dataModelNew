# GPT-5.6 实验结果日志

基线：`submission_v2` 公开榜 **0.70878**，本地 nested OOF **0.69856**。

## S1：冻结融合（探索性）

生成脚本：`scripts/gpt56_frozen_blend.py`

| 候选 | 文件 |
|---|---|
| 50/50 rank mean | `submissions/gpt56_s1_frozen_blend.csv` |
| 75/25 | `submissions/gpt56_s1_v2_75_b7_25.csv` |
| rank max | `submissions/gpt56_s1_rankmax_v2_b7.csv` |

有偏 OOF（B7 含早停乐观，仅诊断）：

| 规则 | OOF |
|---|---:|
| v2 | 0.69910 |
| 75/25 | 0.70145 |
| 50/50 | 0.70285 |
| max | 0.70358 |

**结论：** 仅作探索提交候选；正式晋级前必须诚实重训 B7。

## S2 Stage A：5 折 vs 10 折（cat_d5，seeds 20260/20261）

| 设置 | bagged OOF | seed AUCs |
|---|---:|---|
| 5 folds | 0.69111 | 0.68837, 0.68675 |
| 10 folds | **0.69554** | 0.69293, 0.69270 |

配对结果（`artifacts/gpt56/s2_stage_a_compare.json`）：

| 指标 | 值 |
|---|---:|
| ΔAUC | **+0.00443** |
| 逐 seed Δ | +0.00456, +0.00596 |
| bootstrap 90% CI | [+0.00131, +0.00742] |
| bootstrap 正差比例 | **0.992** |
| Spearman(5f,10f) | 0.974 |

**门槛判定：** 通过（Δ≥0.002 且两 seed 均为正，CI 下界 >0）。

## S2 Stage B：alt 世界 5 折 vs 10 折（cat_alt，seeds 20260/20261）

| 设置 | bagged OOF | seed AUCs |
|---|---:|---|
| 5 folds | 0.68820 | 0.68545, 0.68210 |
| 10 folds | **0.69427** | 0.69368, 0.68833 |

| 指标 | 值 |
|---|---:|
| ΔAUC | **+0.00607** |
| 逐 seed Δ | +0.00823, +0.00622 |
| bootstrap 90% CI | [+0.00267, +0.00950] |
| bootstrap 正差比例 | **0.9985** |

**门槛判定：** 通过。

### 2-seed 融合诊断（仅看配对差值）

| 规则 | OOF |
|---|---:|
| max(d5, alt) @ 5f ×2s | 0.69498 |
| max(d5, alt) @ 10f ×2s | **0.70079** |
| Δ | **+0.00581** |
| 参考：v2 views_max @5f×12s | 0.69910 |

即使只有 2 个种子，10 折融合已超过 12 种子 5 折的 v2。下一步扩到 6–8 种子生产版。

## S2 生产候选：10 折 × 4 seeds（20260–20263）

| 臂 | bagged OOF |
|---|---:|
| cat_d5 | 0.69723 |
| cat_d6 | 0.69677 |
| cat_alt | 0.69631 |

融合（`artifacts/gpt56/v3/fusion_report.json`）：

| 规则 | OOF |
|---|---:|
| views_max | **0.70139** |
| safe 0.75*v3+0.25*v2 | 0.70129 |
| views_half | 0.69922 |
| nested OOF | **0.70147** |
| v2 nested 基线 | 0.69856 |
| **Δ nested** | **+0.00291** |

嵌套选择：5/5 外层块全部选中 `views_max`。

与 v2 `views_max` 的配对 bootstrap（全量 OOF）：

| 指标 | 值 |
|---|---:|
| ΔAUC | +0.00230 |
| 90% CI | [+0.00040, +0.00421] |
| bootstrap 正差比例 | 0.9765 |

提交文件：

- 主候选：`submissions/gpt56_s2_10fold.csv`
- 保守备份：`submissions/gpt56_s2_10fold_safe.csv`
- verify：全部通过

**结论：** 达到正式晋级门槛（nested Δ≥0.002）。建议优先提交 `gpt56_s2_10fold.csv`。

## 代码改动

- `src2/run_oof.py`：`--save-raw` 保存逐 seed/fold 概率与 fold id
- `src2/merge_runs.py`：合并时拼接 raw 数组
- `src2/features.py` / `src2/arms.py`：修复 alt2（保留 `cond_r/ratio`，收缩 local stats）
- `scripts/gpt56_paired_compare.py`：配对 bootstrap
- `scripts/gpt56_frozen_blend.py`：S1 冻结融合
- `requirements.txt`：补 `lightgbm`
