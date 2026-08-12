# 保险理赔二分类：第一性原理重建

## 1. 目标与边界

本方案从 `/workspace/data/{train,test}.csv` 重新建模，以 ROC-AUC 为唯一指标。旧提交、旧融合权重、历史榜单和历史 OOF 不参与候选选择、门禁或最终配方；旧代码只可作为理解字段语义的原料。

核心约束：

1. 所有监督决策必须发生在当前训练分区内部。
2. 外层验证标签只能用于一次最终计分，不能选择特征、方向、超参或融合权重。
3. test 只在方案锁定后生成预测，绝不用于配方选择。
4. 不对全体 train 的弱特征 OOF 做方向翻转，不以大量候选中的最大 OOF 作为结果。

## 2. 数据生成假设

### 2.1 可信信号源

- 理赔风险由车辆/保单状态、时间暴露、地区、来源、等级及其低阶交互共同决定。
- `days`、`condition`、`cc`、`V`、`max_g` 等连续字段可能包含单调或分段关系。
- `month`、`region`、`source`、`version`、`grades`、`code` 等类别字段可能产生稳定组间风险差异。
- 高唯一度 `x0..x18` 可能是连续测量值，也可能只是随机标识的数值化结果，因此必须通过外层验证决定是否保留。
- `id` 是唯一 16 位 hex。默认视为无业务语义标识；只允许测试 label-free 的 byte/nibble 数值表示，禁止目标编码和全局方向选择。

### 2.2 不可信伪信号

- 从全体标签决定 bit/byte/seed 预测是否翻转。
- 在同一 OOF 上反复选择特征子集、权重和配方后仍报告该 OOF。
- 把预计算 OOF 再分成五块求平均并称为 nested。
- 对唯一 ID 生成海量布尔组合，以“低相关”替代真实外层泛化证据。
- 使用 test 预测分布、历史提交或线上反馈挑选配方。

## 3. 模型族

### 3.1 主模型：CatBoost

数据同时包含连续、低基数类别、高基数类别和缺失值。CatBoost 能在不做标签型全局编码的情况下学习非线性及类别交互，适合作为主模型。候选包括：

- Logloss 分类器：直接拟合二分类概率；
- RMSE 回归器：作为损失函数不同的低相关候选；
- depth 5/6、固定树数、强 L2；禁止 outer-valid 早停。

候选超参是有限、预注册集合。每个外折只依据 outer-train 的 inner OOF 选择一个配置。

完整候选最终收敛为两个 RMSE CatBoost：

- `cb_ratio_rmse_d5`：depth 5，800 trees；
- `cb_rate_rmse_d6`：depth 6，800 trees。

每个模型使用 seeds `2026/2027`，预测先做 rank 再平均。三个融合权重 `ratio:rate = 0.35:0.65 / 0.50:0.50 / 0.65:0.35` 在完整运行前锁定；权重不连续搜索，也不使用 outer-valid 拟合。

### 3.2 线性模型

线性/Logistic 模型用于检查稳定的加性信号和作为低复杂度对照，不因单独 AUC 较低而自动融合。只有 outer-train 内选择、outer-valid 增益稳定时才可进入最终方案。

### 3.3 残差与 stacking

第一版不做监督 stacking。后续只有在主模型稳定后，才允许用严格 cross-fit 的一级预测训练二级模型；二级模型和融合权重必须全部在 outer-train 的 inner OOF 上拟合。

## 4. 特征策略

### 4.1 label-free 基础特征

- 原始连续列；
- 原始类别列统一字符串化并显式填充缺失；
- 二值列按类别处理；
- `days`、`condition`、`cc`、`max_g` 的 `log1p(abs(x))` 与符号；
- `days/(1+abs(condition))`、`days*condition` 等少量预注册连续交互；
- `x0..x18` 的逐行均值、标准差、最小、最大和分位数；
- 类别频次与 `days/condition` 的组内均值偏差，统计量只在当前训练分区拟合。

### 4.2 五个预注册特征世界

1. `core`：业务/结构字段，不含 `x0..x18` 和 `id`；
2. `all`：`core` 加原始 `x0..x18` 与行汇总；
3. `all_id`：`all` 加 8 个 byte、16 个 nibble、popcount；仅作消融，默认复杂度最高。
4. `ratio`：`condition` 除以当前训练分区内的 source 中位数，再构造 `days / condition_source_ratio`、折内分位箱及少量类别交互；
5. `rate`：用当前训练分区内每个 source 的 condition 经验分布将 condition 映射为 percentile，再构造 `days * (1-percentile)`、折内分位箱及类别交互。

`ratio/rate` 的 source 统计、经验 CDF、分位边界、类别频率和缺失回退值均在每个 inner/outer 训练分区重新拟合。验证行从不参与这些统计量。

### 4.3 目标编码规则

初版不使用显式目标编码。若后续加入：

- 每个 outer fold 内重新建立；
- outer-train 内用 inner cross-fit 生成训练值；
- outer-valid/test 映射只由对应训练标签拟合；
- 未见类别回退到当前拟合分区先验；
- 方向不得用 outer-valid 或全体 train 标签决定。

## 5. 真正外层 nested 协议

固定算法：

1. 外层 `StratifiedKFold(5, shuffle=True, seed=2026)`。
2. 对每个 outer-train，建立 inner `StratifiedKFold(3)`。
3. 每个候选配置在 inner folds 中重新拟合全部特征统计量和模型，生成完整 inner OOF。
4. 按 inner OOF AUC 选择配置；复杂配置必须比基准高至少 `0.0005`，否则保留简单配置。
5. 选定后在完整 outer-train 重拟合，预测 untouched outer-valid。
6. 汇总五个 outer-valid AUC 的均值、样本标准差及 pooled AUC。
7. 最终训练在全体 train 上重复 inner 选择算法，再拟合 test。

用于研发的 smoke 可减少折数、树数和模型 seeds，但最终推荐必须使用 5 outer folds、至少 3 inner folds和锁定配置集。

## 6. 优化与止损

- 每轮只改变一个轴：特征世界、损失函数或深度。
- smoke 候选若相对同轮基准低 `0.001` 以上，立即停止。
- 完整 nested 晋级门禁：
  - fold-mean 至少提高 `0.001`；
  - smoke 阶段至少 2/3 外折获胜且不得出现明显单折退化；最终确认使用 5 outer folds；
  - pooled AUC 同方向；
  - 置乱标签 AUC 位于 `[0.48, 0.52]`；
  - outer 可在预注册权重间变化，但应稳定选择同一组件族。
- 未过门禁的结果保留在 `artifacts/rebuild/experiments.jsonl`，不生成推荐提交。
- 最终提交仅写 `submissions/submission_rebuild_*.csv`，不覆盖任何旧文件。

## 7. 可复现入口与产物

入口：

```bash
# 广候选 smoke
bash run_rebuild.sh --smoke --permutation-check

# 锁定两个 finalist 的完整 5x3 nested；该命令复现推荐产物
bash run_rebuild.sh --full

# 校验数组长度、有限值、置乱门禁和 submission SHA256
bash run_rebuild.sh --verify
```

当前 `--full` 默认只运行已锁定的 `cb_ratio_rmse_d5,cb_rate_rmse_d6` 和三个预注册 blend。若要重跑历史消融，必须用 `--configs` 显式列出，避免无意扩大最终搜索空间。

产物：

```text
artifacts/rebuild/<run>/
├── metrics.json
├── nested_oof.npy
├── final_inner_oof.npy
├── test_prediction.npy
├── fold_selections.json
├── permutation.json
└── manifest.json
submissions/submission_rebuild_<recipe>.csv
```

## 8. 实验记录

每一行均来自实际运行；`std` 是外折 AUC 的样本标准差。smoke 仅用于淘汰/收敛候选，最终估计只看完整行。

| 实验 | 协议 | fold-mean ± std | pooled AUC | 结论 |
|---|---|---:|---:|---|
| 原始/core/all/id 广候选 | 3 outer × 2 inner，1 seed，250/300 trees | 0.668683 ± 0.006066 | 0.668679 | `all_id` 最差；置乱 0.502032 |
| ratio/rate 机制消融 | 3×2，1 seed，250/300 trees | 0.673806 ± 0.012142 | 0.673804 | 固定 rate 0.678030，进入下一轮 |
| ratio/rate blend | 3×2，1 seed，300 trees | 0.679815 ± 0.008104 | 0.679858 | 最佳固定 blend 0.680300，较单臂 +0.002270 |
| **最终选择算法** | **5 outer × 3 inner，2 seeds，800 trees** | **0.688722 ± 0.012695** | **0.688704** | 5/5 外折均选择 ratio/rate blend |

最终五折 AUC 为：

```text
0.69695661, 0.67897333, 0.68956813, 0.67355084, 0.70456317
```

完整运行中的外折选择依次为 `w50 / w35 / w65 / w50 / w65`。这说明“两个机制需要融合”稳定，但精确权重仍有抽样不确定性。最终全训练集 inner 选择为 `w65`，inner AUC `0.683282`；该数仅用于最终配方选择，不作为无偏性能估计。

置乱哨兵在独立 3-fold outer 上得到 `0.492784 ± 0.007624`，pooled `0.492789`，通过 `[0.48, 0.52]` 门禁。

## 9. 最终产物

- 无偏性能估计：outer nested `0.688722 ± 0.012695`，pooled `0.688704`；
- 最终配方：`0.65 * cb_ratio_rmse_d5 + 0.35 * cb_rate_rmse_d6` 的 rank blend；
- 推荐文件：`submissions/submission_rebuild_blend_ratio_rate_w65.csv`；
- SHA256：`787c4fd456a53b6e63f297e9d2ad84137f386ba79e92c58b0c1ca11d1d8ddaa2`；
- 完整证据：`artifacts/rebuild/full/metrics.json`；
- 复核命令：`bash run_rebuild.sh --verify`。

## 10. 已证伪与下一轮

本轮已证伪或停止：

- `id` byte/nibble：固定 outer smoke `0.665772`，明显低于不含 ID 的 core；停止。
- 原始 `x0..x18`：同深度 Logloss 下 `all=0.669431`、`core=0.669624`，没有增益；停止直接喂入和行汇总扩展。
- ratio-Logloss：`0.676910`，低于 ratio-RMSE `0.677695`；不进入完整运行。
- 无监督 stacking、全局方向翻转、显式目标编码和大规模 ID 组合搜索：没有泄漏安全且稳定的晋级证据，不做。

仍值得继续、但必须使用新的预注册确认实验：

1. 将 seeds 从 2 增至 4，只测试已经锁定的 ratio/rate/w65，使用新的 outer seed 作稳定性确认；
2. 对两个一级模型做严格 cross-fit 的低自由度 Logistic stacking，权重/正则只在 inner OOF 拟合；
3. 为 source 经验 CDF 增加 shrinkage 与 leave-one-out 的 label-free 稳健版本，防止小组分位跳变；
4. 报告多个 outer seeds 的均值和 seed 间方差，但不得据此反复改特征后仍把同一批 seeds 当最终无偏估计。
