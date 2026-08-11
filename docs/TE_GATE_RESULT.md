# TE 第三臂门槛结果（完整 8×5×3）

命令：`bash run_super714.sh`（8 seeds × 5 folds × 3 bags，耗时 93.1 分钟）

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---|:---:|
| TE OOF AUC | **0.69964** | > 0.697 | ✅ |
| Spearman(TE, main) | **0.99767** | < 0.90 | ❌ |
| max3 − max2 | **−0.00018** | > 0.001 | ❌ |

**结论：拒绝 TE 第三臂；主交付保持 best_v1 max2。**

解读：TE 被 CatBoost 主臂 CTR / `d10s` 交叉完全吸收，与 main 几乎共线（Spearman≈0.998），max3 无增益。与数据挖掘预告一致。

证据：`artifacts/super714/metrics.json`、`artifacts/super714/main_te.npz`

## Smoke（历史通路检查）

| 指标 | 结果 |
|---|---:|
| TE OOF | 0.67661 |
| Spearman(TE, main) | 0.88223 |
| max3 − max2 | −0.00371 |

Smoke 未覆盖主提交；最终判定以本页完整门槛为准。
