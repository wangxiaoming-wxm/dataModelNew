# TE 第三臂门槛结果

## Smoke（通路检查，非最终判定）

命令：`bash run_super714.sh --smoke`（2 folds × 1 seed × 1 bag）

| 指标 | 结果 | 完整门槛 | smoke 观测 |
|---|---:|---|---|
| TE OOF AUC | 0.67661 | > 0.697 | 未达 |
| Spearman(TE, main) | 0.88223 | < 0.90 | 达到 |
| max3 − max2 | −0.00371 | > 0.001 | 未达 |

Smoke **不会覆盖**主提交（代码强制 `accepted = (not smoke) and gates`）。  
**完整 8×5×3 门槛仍在运行/待产出 `artifacts/super714/metrics.json`；在完整结果落盘前，不得宣称 TE 已被最终拒绝或接受。**

## 补充探针（信息性，不替代门槛）

- 2seed×5fold×2bag×400iter，main+折内 TE：AUC=0.68757，corr(main)=0.9466，max3 Δ=−0.00143
- TE-lite 弱特征：AUC=0.656，max3 Δ=−0.0079

证据：`artifacts/super714/metrics_smoke.json`
