# 独立审核：rebuild V2（submission_rebuild_blend_rich_ratio_rate_w65）

审核人：接管 agent（新分支 `cursor/vz20-championship-edb2`）。  
被审方案：子 agent 冻结的 V2，诚实 nested **0.695181 ± 0.012759**。

## 裁决

**不作为夺冠或冲榜主交。** 方案协议诚实，但绝对水平不够，且弱于已上线的 W62。

## 靠谱的部分

- 评价器用了外层折内拟合、折内选权重、置乱检验（≈0.495），没有 fp_v8 那种全标签翻转。
- V3 四个正交方向（频率特征、交叉深度、nested residual、Logloss 多样性）均未过 `+0.001` 门禁，止损合理。
- 代码可复现：`bash run_rebuild.sh --verify`。

## 不靠谱 / 不够用的部分

1. **绝对分数**：诚实 nested 0.695 < 冻结 W62 OOF 0.70159。同一指标下已经输给 2026-08-11 的双世界 CatBoost。
2. **线上外推**：W62 用 0.70159 本地拿到 0.71503。V2 若 gap 相似，期望线上约 0.708，低于 W62，更低于 0.72/0.749。
3. **特征世界**：V2 的 ratio/rate rich 与 SUPER714 `build_main`/`build_alt` 同源。子 agent 自己也证明再堆 rich/freq/residual 边际 <0.001。
4. **训练预算**：V2 用 4 seeds×800 trees，W62 是 8 seeds×5 folds×3 bags×800。V2 并没有在同构轴上把已知最强配方跑满。
5. **夺冠鸿沟**：0.749 − 0.715 ≈ **+0.034**。这不是 bagging、blend 权重或 byte TE 能磨出来的；需要尚未被双世界 CatBoost 吃到的强信号。本轮未找到。

## 因此怎么用 V2

- 保留为「诚实 nested 管线」样本，不要覆盖 W62。
- 后续实验若声称超过 W62，必须在**同一外层划分**上同时报告 W62/AM40 对照。
