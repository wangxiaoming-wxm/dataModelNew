# Rebuild V2 预注册评价门禁

本文件在 V2 实现和训练前定义，不根据后续 outer-valid 结果修改阈值。

## 能力验证 A：rich ratio/rate

候选只增加 label-free、fold-fitted 的分位类别交互，并恢复与模型机制匹配的 boosting 类型：

- `ratio_rich`：source-condition 尺度世界，Ordered depth-5 RMSE；
- `rate_rich`：source-condition 经验 CDF 世界，Plain depth-6 RMSE。

3 outer × 2 inner smoke 晋级条件：

1. 至少一个 rich 单臂相对对应 simple 单臂 fold-mean 提高 `>= 0.001`；
2. 至少 2/3 外折获胜；
3. 任一外折退化不得超过 `0.005`；
4. rich blend 相对最好 rich 单臂提高 `>= 0.0005` 才保留 blend 搜索。

## 能力验证 B：方差降低

只有能力 A 晋级后才测试。锁定特征、模型和权重，增加 model seeds，不改变 outer/inner 划分。

晋级条件：

1. 4-seed 相对 1-seed smoke fold-mean 提高 `>= 0.0005`；
2. 至少 2/3 外折不退化；
3. 训练代价记录在 metrics 中。

## 能力验证 C：严格 stacking

仅允许两个一级模型的 cross-fitted 预测。元模型只能在 outer-train 的 inner OOF 上拟合，outer-valid 不得参与权重、符号或正则选择。

晋级条件：

1. 相对同轮最佳固定 blend 提高 `>= 0.001`；
2. 至少 2/3 smoke 外折获胜；
3. 元模型系数在各 outer folds 中符号一致且不过度极端。

若 B 已过门禁且算力接近预算上限，可以停止 C，避免在同一 outer seed 上扩大实验者搜索空间。

## 完整 V2 门禁

最终候选使用 5 outer × 3 inner；所有特征统计在训练分区拟合。相对 V1 的 `0.688722`：

- fold-mean 提高 `>= 0.001`；
- pooled AUC 提高 `>= 0.001`；
- 至少 3/5 外折高于 V1 对应折；
- permutation fold-mean 位于 `[0.48, 0.52]`；
- 五个 outer folds 选择同一模型族，不要求精确 blend 权重相同。

若未通过，V1 保持推荐；不得因为 final-inner 分数较高而替换。

## 回归验证

- `python3 -m unittest tests.test_rebuild -v`
- `bash run_rebuild.sh --verify`
- 新 V2 产物必须通过同等长度、有限值、SHA256 和置乱哨兵校验。
