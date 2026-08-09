# S6：最终模型选择、提交预算与回滚

## 1. 原则

最终目标不是提交本地分最高的文件，而是提交“最可能在未知测试标签上超过 0.70878”的文件。
候选必须同时满足：

- 强度：配对 OOF 确实提升；
- 稳定：不同 seed、外层块和 bootstrap 下方向一致；
- 多样：新增臂提供互补，而不是同一模型的噪声副本；
- 可追溯：代码、配置、数据和产物都有 hash；
- 克制：不通过公开榜反复调整权重。

## 2. 候选池

最多保留以下 6 类，每类只留一个冠军：

| ID | 候选 | 来源 |
|---|---|---|
| C0 | v2 冻结基线 | `submission_v2.csv` |
| C1 | v2 + B7 rank mean | S1 |
| C2 | 10 折升级版 | S2 |
| C3 | 新编码世界融合 | S3 |
| C4 | 结构 bagging | S4 |
| C5 | 正交臂融合 | S5 |

如果某阶段未通过门槛，对应候选为空。不能为了凑满候选降低门槛。

## 3. 先做 Pareto 淘汰

每个候选计算：

```text
nested_oof_auc
delta_vs_v2
paired_ci90
paired_ci95
probability_delta_gt_0
positive_seed_ratio
outer_block_wins
spearman_vs_v2
test_spearman_vs_v2
training_cost
pipeline_complexity
```

若候选 A 同时满足：

- OOF 不低于 B；
- 置信区间不差于 B；
- 复杂度/成本不高于 B；
- test 排序变化不比 B 更极端；

则 B 被 A Pareto 支配并淘汰。不要把被支配候选放进融合“增加多样性”。

## 4. 最终统计评估

### 4.1 seeds 只能验证随机稳定性

discovery/confirmation seeds 使用的是同一批标签，它们能防止“碰巧选中一个好随机种子”，
但不能完全消除在同一组样本上筛很多候选造成的选择乐观。因此最终主报数字必须再做按行嵌套评估。

### 4.2 按行嵌套选择

在开始最终融合前冻结候选和最多 6 条规则。使用固定的 5 个 stratified meta-fold：

1. 在 4 个 meta-fold 上选择候选成员与规则；
2. 在第 5 个 meta-fold 上应用选择结果；
3. 循环 5 次拼成完整 nested OOF；
4. held-out meta-fold 从未参与该块的成员/规则选择。

所有基础预测本身仍必须是训练折外 OOF。meta-fold 只是选择层，不替代模型 CV。

报告：

- nested OOF AUC；
- 5 个 held-out 块各自的 ΔAUC；
- 每个规则的选择次数；
- 拼接预测相对 v2 的配对 bootstrap；
- 全量最优与 nested 结果的差距。

如果全量 OOF 很高、nested OOF 回落 >0.002，说明融合选择过拟合，退回更简单规则。

## 5. 最多 6 条预登记最终规则

根据实际通过的候选，固定：

```text
R0 = v2
R1 = best_single_upgrade
R2 = 0.50*v2 + 0.50*best_single_upgrade
R3 = 0.75*v2 + 0.25*best_diversity_arm
R4 = equal_rank_mean(all qualified strong arms)
R5 = rank_max(all arms whose standalone gap <= 0.0015)
```

限制：

- 弱臂不进入 R5；
- 不对连续权重做优化；
- 每类候选只能贡献一个版本；
- B7 若已在 C1 内，不作为第二个独立臂重复计权；
- 全部输入先转成 `[0,1]` 的经验秩。

最终规则：

- 5 块中同一规则胜出 ≥4 次：用该规则；
- 选择分散但 R2/R4 始终在前二：用二者平均；
- 没有任何升级规则稳定胜过 R0：继续提交 v2，不强行换版本。

## 6. 最终晋级门槛

一个新文件要替代 v2，至少满足：

| 项 | 门槛 |
|---|---:|
| nested ΔAUC | ≥ +0.002 |
| 配对 bootstrap `P(Δ>0)` | ≥ 95% |
| 90% CI 下界 | > 0 |
| held-out meta-fold 胜出 | ≥ 4/5 |
| 正向 seeds | ≥ 75% |
| test Spearman vs v2 | ≥ 0.94 |
| shuffled-label AUC | 0.47～0.53 |

对于与 v2 相关 ≤0.92 的真正正交臂，可将 nested Δ 门槛放宽到 +0.0015，但必须使用小权重 mean，
不能直接替代主模型。

## 7. 提交预算

先确认比赛实际剩余提交次数；以下按“至少 3 次可用”规划：

### 提交 1：低成本结构确认

`C1` 的 v2+B7 50/50 rank mean。目的仅是利用两套已经冻结且各自有真实榜单结果的系统。
无论结果如何，不调整权重。

### 提交 2：本地冠军

S2～S5 完成后，提交 nested 规则选出的最高期望候选。这是主提交。

### 提交 3：保守备份

仅当主冠军与 v2 test Spearman <0.97 时，提交：

```text
0.75 * rank(main_champion) + 0.25 * rank(v2)
```

该规则必须提前登记，不根据提交 2 的榜单分修改比例。

如果只剩 1 次：跳过 C1，直接交本地冠军。

如果只剩 2 次：交本地冠军和预登记保守备份，不做榜单权重搜索。

## 8. 榜单结果使用边界

允许：

- 记录分数和提交文件 hash；
- 判断文件是否被平台正确解析；
- 在比赛结束后分析 CV-LB 关系。

不允许：

- 因某次上涨把相邻权重再细扫一遍；
- 因某次下降删除本地稳定有效的特征；
- 用两三个榜单点拟合“本地分→榜单分”回归；
- 从多个近似提交中挑公开榜最高者再继续迭代。

单次半榜 AUC 的噪声约 0.015，通常大于本轮期望模型增益。公开榜只能做最终确认，不能做训练标签。

## 9. 提交前流水线

```bash
# 1. 数据与环境
sha256sum -c data/SHA256SUMS.txt
git status --short

# 2. 复算最终融合（命令按最终 artifact 调整）
PYTHONPATH=src2 python3 src2/fuse.py \
  --dir artifacts/gpt56/final \
  --submission submissions/gpt56_final.csv

# 3. 完整检查
PYTHONPATH=src2 python3 src2/verify.py \
  --submission submissions/gpt56_final.csv

# 4. 文件检查
sha256sum submissions/gpt56_final.csv
wc -l submissions/gpt56_final.csv
```

额外断言：

```text
columns == ["id", "label"]
rows == 6398
id order == submit_sample.id
label finite
0 <= label <= 1
n_unique(label) > 1000
```

最终 report 中保存所有父预测 hash，避免提交文件与报告错配。

## 10. 命名与产物

```text
submissions/
  gpt56_s1_frozen_blend.csv
  gpt56_champion.csv
  gpt56_champion_safe.csv

artifacts/gpt56/final/
  candidate_table.csv
  nested_selection.json
  bootstrap_delta.json
  prediction_correlations.csv
  manifest.json
  verify.json
```

`manifest.json` 应标明：

```text
baseline_lb = 0.70878
baseline_file_sha256
candidate_file_sha256
git_commit
model_artifact_sha256s
selected_rule
selection_picks
no_test_labels = true
leaderboard_not_used_for_tuning = true
```

## 11. 回滚

- 训练中断：保留已完成 seed 段，用 manifest 判断能否等权合并；模型数不相等时按模型数加权；
- 新世界确认失败：回滚到 v2/S2，不改旧产物；
- nested 选择不稳定：回滚到 R2 的简单平均；
- 完整候选不达晋级门槛：继续使用 `submission_v2.csv`；
- 提交格式失败：不修改预测值，只修复 ID/列名/行序后重新校验。

## 12. 建议执行日程

| 顺序 | 工作 | 停止点 |
|---:|---|---|
| 1 | S0 仪器与基线 | 基线不能精确复算则不继续 |
| 2 | S1 冻结融合 | 最多消耗 1 次提交 |
| 3 | S2 2-seed 5/10 折 | Δ<0.001 立即停 |
| 4 | S3 alt2-repair + 8～12 世界 discovery | 最多 4 个晋级 |
| 5 | S3 confirmation | 最多保留 2～3 个 |
| 6 | S4 E1→E2→E3 | 只保留独立通过项 |
| 7 | S5 快速备用路线 | <0.685 停 |
| 8 | S6 嵌套选择与最终提交 | 不达门槛则保留 v2 |

这套顺序把低成本、高证据的动作放在前面，并保证任何阶段失败都能安全回到已知的 `0.70878` 基线。

