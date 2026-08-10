# V4max3proNew

## 一句话

参考声称公开榜 **0.71504** 的 `715.zip`（语义三阶交叉 FE + CatBoostRegressor RMSE + 10 bagging），在本仓冻结 max3 / v4max3pro 尺子下做增量融合，产出 `v4max3proNew`。

**不直接宣称本提交会到 0.71504**：zip 内是单臂本地 ~0.695 量级 OOF，未见其融合与提交文件；本仓以嵌套 5-block AUC 诚实准入。

## 来源臂（忠实移植）

| 项 | 值 |
|---|---|
| FE | `src4/feat_semantic.py`（raw + structured_string + days_condition + dual_category TOP_CROSS） |
| 模型 | CatBoostRegressor `RMSE`，depth=6，lr=0.03，iter=900，ES=120 |
| 协议 | 5 seed × 5 fold × 10 bagging，折内 `fit` |
| 产物 | `artifacts/v4max3pronew/semantic_rmse.npz` |

```bash
python3 src4/train_semantic_rmse.py --smoke
python3 src4/train_semantic_rmse.py --seed 2026   # ×5 seeds
python3 src4/train_semantic_rmse.py --merge-only
```

## 融合与准入

```bash
python3 src4/build_submission_v4max3pronew.py          # 评估
python3 src4/build_submission_v4max3pronew.py --write  # 仅当准入通过或 exploratory 条件满足
python3 src4/build_submission_v4max3pronew.py --check
```

候选族：
- `max(max3, semantic_rmse)`
- `max(v4max3pro, semantic_rmse)`
- `max(max3, plus_strong, semantic_rmse)`（去掉与 ord_noxb 共线的 noxb10）
- 对应 `rmean` 对照

准入：嵌套 AUC 超过 max3 且不低于 pro，且相对 `submission_v4_max3.csv` 的 Spearman < 0.9995。

正式提交（若准入）：`submissions/submission_v4max3pronew.csv`

## 风险

1. ES + 多 bagging 抬高本地 OOF（PROTOCOL_RISK）。
2. 单臂 OOF 低于 max3 嵌套时，增益完全依赖 rank-max 互补；可能本地涨、线上不涨。
3. 715 声称 0.71504 无法在本仓复现核验（无其 CSV / 无其融合脚本）。
