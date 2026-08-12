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

### 4.2 三个预注册特征世界

1. `core`：业务/结构字段，不含 `x0..x18` 和 `id`；
2. `all`：`core` 加原始 `x0..x18` 与行汇总；
3. `all_id`：`all` 加 8 个 byte、16 个 nibble、popcount；仅作消融，默认复杂度最高。

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
  - 至少 4/5 外折获胜；
  - pooled AUC 同方向；
  - 置乱标签 AUC 位于 `[0.48, 0.52]`；
  - outer 选择结果不过度分散。
- 未过门禁的结果保留在 `artifacts/rebuild/experiments.jsonl`，不生成推荐提交。
- 最终提交仅写 `submissions/submission_rebuild_*.csv`，不覆盖任何旧文件。

## 7. 可复现入口与产物

计划入口：

```bash
bash run_rebuild.sh --smoke
bash run_rebuild.sh --full
bash run_rebuild.sh --verify
```

产物：

```text
artifacts/rebuild/<run>/
├── metrics.json
├── nested_oof.npy
├── final_inner_oof.npy
├── test_prediction.npy
├── fold_selections.json
└── manifest.json
submissions/submission_rebuild_<recipe>.csv
```

## 8. 实验记录

本节在实际训练后补充，每一行必须来自真实外层 nested 运行。

| 实验 | 协议 | fold-mean ± std | pooled AUC | 结论 |
|---|---|---:|---:|---|
| 待运行 | — | — | — | — |
