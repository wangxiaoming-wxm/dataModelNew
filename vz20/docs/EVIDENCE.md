# vz20 实验证据

所有数字来自 `artifacts/vz20/metrics.json`（outer seed 90210, 5 折, 4 seeds, 800 trees），
可由 `python3 src/vz20_combine.py --profile full --outer-splits 5` 复算（读缓存，秒级）。

## 1. 主门禁裁决

| 项 | 值 | 门禁 | 结果 |
|---|---|---|---|
| fold-mean lift (vz20 − vz19) | **+0.00021** | ≥ +0.0015 | ❌ FAIL |
| 不退化外折 | **2/5** | ≥ 3/5 | ❌ FAIL |
| 置乱 fold-mean / pooled | 0.5128 / 0.5129 | ∈[0.48,0.52] | ✅ PASS |

**总裁决：门禁未过。vz20 与 vz19 在统计上不可区分。**

## 2. 逐折对比（配对）

| fold | vz19 | vz20 | diff |
|---|---|---|---|
| 0 | 0.71734 | 0.71732 | −0.00002 |
| 1 | 0.67316 | 0.67244 | −0.00072 |
| 2 | 0.70464 | 0.70368 | −0.00096 |
| 3 | 0.69605 | 0.69677 | +0.00072 |
| 4 | 0.70666 | 0.70867 | +0.00201 |

正负相抵，均值 +0.0002，落在噪声内（折间 std≈0.015）。

## 3. 组件分解（关键证据）

| 组件 | fold-mean | 增量 |
|---|---|---|
| my_cb (A1+A2) | 0.69889 | — |
| ref (REF1+REF2) | 0.69813 | — |
| max2(my_cb, ref) | 0.69882 | **−0.00007**（max2 在新 seed 上≈无效） |
| vz19 = max2 + byte07 | 0.69957 | **+0.00075**（byte07 是唯一真实正交增益） |

> **重要发现**：vz19 文档声称 max2 带来 +0.00176。本 harness 在**从未参与调参的 outer seed** 上
> 测得 max2 增益 = **−0.00007**。这实证了上一轮审核的判断：max2 的 OOF 增益来自"对同一份被污染
> OOF 的重采样"，不是可泛化收益。byte07（cross-half 已验证）则在新 seed 上稳定贡献 +0.00075。

## 4. 正交性证据：rich ≈ vz19 的重复

各臂对 vz19 arm1(A1) 的 Spearman（fold0）：

| 臂 | Spearman vs A1 | 解读 |
|---|---|---|
| R1 ratio_rich | **0.9845** | 几乎是 A1 的复制 |
| R3 ratio_freq | 0.9831 | 同上 |
| REF1 main-Plain | 0.9678 | 同世界不同 boosting |
| R2 rate_rich | 0.9254 | rate 世界（vz19 已由 A2 覆盖） |
| R4 rate_freq | 0.9208 | 同上 |
| A2 alt | 0.9161 | rate 世界 |
| REF2 alt-Ordered | 0.9169 | rate 世界 |

rich/freq 的 ratio 实现与 vz19 arm1 相关性 0.98+，几乎不带新信息；rate 世界的多样性
（~0.92）vz19 早已用 arm2 吃到。**没有可榨的正交信号。**

## 5. 融合天花板（oracle 上界，非法但用于定性）

即使**在 eval 上直接调**融合权重（作弊，非门禁合法），vz19 与 8-arm MEGA 的最优凸组合
fold-mean = 0.69977 @ w=0.95（即基本就是 vz19），lift 仅 **+0.0002**。
→ **门禁 +0.0015 在现有信号下不可达。**

## 6. 诊断备选配方（均未过门禁，仅记录）

| 配方 | lift vs vz19 |
|---|---|
| vz20 主（3 实现家族） | +0.00021 |
| vz20_4arm（rich, 不含 freq） | +0.00034 |
| vz20_max3（并入 rich 作第三 max） | +0.00041 |
| MEGA 8-arm | +0.00020 |

## 7. 证伪 / 未采纳（诚实披露）
- 未加深 id 字节（byte6/3 已证伪）；未碰 fp_v8/championship 0.77 泄漏旁路。
- 未启用 stacking：两方案 Spearman 0.98，meta-model 无多样性可用（rebuild 也证伪）。
- 未在 outer-valid 上搜权重；未用 test 标签/历史线上/伪标签。
- 未把 x0..x18 直接喂入或做行汇总（rebuild 已证无增益）。
