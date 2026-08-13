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

4-seed 通过后允许一次预注册的 8-seed 止损验证：seeds 固定为 `2026..2033`，相对 4-seed fold-mean 必须再提高 `>= 0.0003`，且至少 2/3 外折不退化。未通过则固定 4-seed；通过才允许将完整 V2 扩到 8-seed。不得继续搜索 seed 子集。

## 能力验证 C：严格 stacking

仅允许两个一级模型的 cross-fitted 预测。元模型只能在 outer-train 的 inner OOF 上拟合，outer-valid 不得参与权重、符号或正则选择。

唯一预注册元模型为 `LogisticRegression(C=0.1, L2, lbfgs)`，输入是 rich ratio/rate 的 rank 预测；C 不搜索。用于 inner 计分的元模型本身再次按 inner folds cross-fit，outer-valid 预测则由完整 outer-train inner OOF 拟合元模型后生成。

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

## 冻结结果

阈值定义后得到：

| 验证 | 结果 | 裁决 |
|---|---:|---|
| rich ratio 单臂 vs simple | `+0.005885`，3/3 外折提升 | 通过 |
| rich rate 单臂 vs simple | `+0.006343`，3/3 外折提升 | 通过 |
| rich w50 vs 最好 rich 单臂 | `+0.002277` | 通过 |
| 4-seed vs 1-seed rich w50 | `+0.002341`，3/3 提升 | 通过 |
| strict stack vs 4-seed w50 | `+0.000017` | 淘汰 |
| 8-seed vs 4-seed 选择算法 | `+0.000211` | 低于 `+0.0003`，淘汰 |
| V2 full fold-mean vs V1 | `+0.006459`，5/5 提升 | 通过 |
| V2 full pooled vs V1 | `+0.006396` | 通过 |
| permutation | `0.495288 ± 0.003677` | 通过 |

最终冻结为 rich ratio/rate、4 seeds、inner 选择 `w35/w50/w65`。完整 outer nested 为 `0.695181 ± 0.012759`，pooled `0.695101`。
