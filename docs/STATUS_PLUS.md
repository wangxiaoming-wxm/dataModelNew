# 进展看板（相对 W62）

| 方案 | 本地 OOF | 线上 AUC | 决策 |
|---|---|---|---|
| best_v1 max2 | 0.70128 | 0.71453 | 勿再交同文件 |
| W62 | 0.70159366 | **0.71503** | 旧最强锚点 |
| **AM40** | **0.70181135** | — | **当前地板：已超过 W62（Δ=+0.00022）** |
| Plus / Extend | ≤0.70125 | — | 拒绝 |
| SUPER714-Bags (6bag 同构) | 训练中 | — | 若更高则晋升 `submission_beat_w62.csv` |

## 当前可提交

- `submissions/submission_am40.csv`
- `submissions/submission_beat_w62.csv`（= AM40，门禁已过）
- 复核：`bash run_am40.sh --verify`

## Bags 训练

- 日志：`logs/super714_bags_full.log`
- 完成后自动比较；仅当 OOF > W62 才覆盖 beat 文件
