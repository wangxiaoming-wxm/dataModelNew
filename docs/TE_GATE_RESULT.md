# TE 第三臂门槛结果

## Smoke（预注册门槛，未放宽）

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---|:---:|
| TE OOF AUC | 0.67661 | > 0.697 | ❌ |
| Spearman(TE, main) | 0.88223 | < 0.90 | ✅ |
| max3 − max2 | −0.00371 | > 0.001 | ❌ |

结论：**拒绝 TE 第三臂**；主交付保持 best_v1 max2。

## 补充探针（父进程，非放宽门槛）

- 2seed×5fold×2bag×400iter，main 特征 + 折内平滑 TE：AUC=0.68757，corr(main)=0.9466，max3 Δ=−0.00143
- TE-lite 弱特征：AUC=0.656，max3 Δ=−0.0079
- 历史高相关旧臂（b7/noxb）本地 max3 虚高，但其真实 LB ≤ 0.71222，禁止并入

证据文件：`artifacts/super714/metrics_smoke.json`、`artifacts/probe/te_main_probe.npy`
