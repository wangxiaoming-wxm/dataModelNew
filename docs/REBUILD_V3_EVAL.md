# Rebuild V3 预注册评价协议

本文件在查看新 outer seed 结果和实现 V3 候选前冻结。

## Phase S：V2 跨 seed 稳定性

锁定 V2，不修改特征、模型、权重集合、树数或 model seeds：

- outer：`StratifiedKFold(5, seed=314159)`；
- inner：`StratifiedKFold(3, base_seed=161803)`；
- model seeds：`2026..2029`；
- 候选：rich ratio/rate 与 `w35/w50/w65`；
- 800 trees，完整 fold-fitted 统计；
- 同时运行 permutation sentinel。

稳定门禁：

1. fold-mean `>= 0.685`；
2. pooled AUC `>= 0.685`；
3. permutation fold-mean 在 `[0.48, 0.52]`；
4. 5/5 outer folds 仍选择 rich 双臂族。

Phase S 只确认稳定性，不用其结果修改 V2。

## Phase A：类别可靠性频率

假设：高阶分位交互的类别值有不同支持度；CatBoost 只看到类别 token，未显式知道该 token 是常见组还是小样本组。V3 为预注册交互增加纯 label-free 的训练分区频率和 rare 指示：

- 频率表只在当前 inner/outer 训练分区拟合；
- valid/test 未见值映射为 0；
- rare 阈值固定为训练分区 `5 / n_train`；
- 不使用标签、test 分布或全局频率。

新 seed `3 outer × 2 inner × 4 model seeds × 300 trees` smoke 门禁：

1. 至少一个 frequency 单臂相对对应 rich 单臂 fold-mean `+0.001`；
2. 至少 2/3 对应外折提升；
3. 任一外折不得退化超过 `0.004`；
4. frequency blend 相对新 seed V2 最佳固定 blend至少 `+0.001`。

未过即淘汰，不微调频率列或 rare 阈值。

## Phase B：交叉深度

只有 Phase A 未通过或增益不足时启用。预注册候选：

- ratio-rich Ordered depth 6；
- rate-rich Plain depth 5。

仍使用新 seed smoke。单臂或四模型固定 rank mean 必须相对新 seed V2 最佳 blend提高 `>= 0.001` 且至少 2/3 外折提升，才进入完整确认。

## Phase C：nested-nested 残差

仅当 A/B 均失败时考虑。残差训练目标必须来自 sub-inner cross-fit 的 base prediction；禁止用 in-sample base residual。残差权重只能在 outer-train 的 inner OOF 上选择。

由于该方案每个 inner fold 还需一层 sub-inner，只有预估模型训练数可控且 A/B 无晋级方案时实现。晋级阈值仍为 smoke `+0.001`、至少 2/3 外折提升。

冻结实现：

- base 为 rich ratio/rate 的固定 `w50` rank blend；
- 对每个 inner-train 再做 `StratifiedKFold(2, seed=424243+fold)`，生成严格 sub-inner base OOF；
- residual target 固定为 `y - base_oof_rank`；
- residual learner 固定为 Plain core-RMSE depth 5；
- outer-valid 组合固定为 `base_rank + 0.20 * residual_raw`，不搜索 alpha、不翻转方向；
- residual learner 的训练分区不包含对应 inner-valid/outer-valid。

## 完整确认与晋级

开发仅看新 seed smoke。候选锁定后，在原 seed `2026/2718` 上运行完整 `5 outer × 3 inner × 4 model seeds × 800 trees`：

- 相对 V2 `0.695181` 的 fold-mean 和 pooled AUC 均提高 `>= 0.001`；
- 至少 3/5 对应外折提升；
- permutation 继续通过；
- 最终才生成新 submission。

若完整确认失败，推荐保持 V2；不得依据 final-inner 或 test 预测分布替换。
