# SUPER714-Plus 进展（相对 W62）

| 方案 | 本地 OOF | 线上 AUC | 状态 |
|---|---|---|---|
| best_v1 max2 | 0.70128 | 0.71453 | 公开冠军锚点；勿再交同文件 |
| **W62** `0.62*main+0.38*alt` | **0.70159** | **0.71503** | **当前已提交最强**（`task_w62`） |
| SUPER714-Plus max2 / w62 / wbest | 训练中 | — | `cursor/super714-plus-edb2` |

## 训练进度（全量）

- 命令：`python3 -u src_super/train_super714_plus.py`
- 日志：`logs/super714_plus_full.log`
- 协议：10 seeds (3100–3109) × 4 bags × 1000 trees；main Ordered d6，alt Plain d6
- 结束后执行：`python3 -u src_super/fuse_plus_weights.py`

## 下一步（训练完成后）

1. 读 `artifacts/super714_plus/metrics.json` 与 `fuse_weights_metrics.json`
2. 优先比较 Plus-w62 / Plus-wbest / Plus-max2 相对 W62 的 OOF 与 test Spearman
3. 若 Plus OOF 不低于 best_v1，优先提交 **Plus-w62**（预注册权重，跟线上证据一致）
4. 若 Plus 明显弱于 W62，保留 W62 为最强，另开更轻的差异化实验（勿盲目交 wbest）
