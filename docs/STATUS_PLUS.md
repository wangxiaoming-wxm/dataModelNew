# 进展看板（相对 W62）

| 方案 | 本地 OOF | 线上 AUC | 决策 |
|---|---|---|---|
| best_v1 max2 | 0.70128 | 0.71453 | 公开锚点；勿再交同文件 |
| **W62** `0.62*main+0.38*alt` | **0.70159** | **0.71503** | **当前最强，继续持有** |
| Plus max2 / w62 / wbest | ≤0.69803 | — | **拒绝**（改 depth/分箱/跨世界伤害信号） |
| Extend-w62 `(8*frozen+4*new)/12` | 0.70125 | — | **拒绝**（新 seed 更弱，稀释冻结臂） |

## Plus（已完成）

- max2 0.69645，Δ vs best_v1 = **−0.00483**
- 与 W62 混合最优权重 = 1.0（无互补）
- 产物：`submissions/submission_super714_plus*.csv`

## Extend（已完成）

- 新 seed 2034–2037 同构续训；new main/alt pool = 0.69763 / 0.69404（低于冻结）
- merged_w62 = 0.70125，相对冻结 W62 **−0.00034**
- Spearman(extend_w62, submission_w62) ≈ 0.9999（几乎同排序，仍略弱）
- 产物：`submissions/submission_super714_extend*.csv`，`artifacts/super714_extend/metrics.json`

## 结论

1. **主提交继续用 W62（线上 0.71503）**  
2. Plus / Extend 均未提供可提交增益，作负结果存档  
3. 下一步若再冲分：需要**新信号轴**或更稳的同构增强（例如同 seed 加 bag 全量重训并设 OOF 门禁），而不是盲加弱 seed / 改世界定义
