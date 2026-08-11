# SUPER714-Plus 结果（相对 W62）

| 方案 | 本地 OOF | 线上 AUC | 状态 |
|---|---|---|---|
| best_v1 max2 | 0.70128 | 0.71453 | 公开冠军锚点；勿再交同文件 |
| **W62** `0.62*main+0.38*alt` | **0.70159** | **0.71503** | **当前已提交最强**（`task_w62`） |
| Plus max2 | 0.69645 | — | **拒绝**：ΔOOF vs best_v1 = **−0.00483** |
| Plus-w62 | 0.69754 | — | 仍低于 W62 |
| Plus-wbest (w=0.80) | 0.69803 | — | 仍低于 W62；不建议交 |

## 全量训练结论

- 协议：10 seeds (3100–3109) × 4 bags × 1000 trees；main Ordered d6 +rate；alt Plain d6 +ratio/cond_r bins(6,12,24)
- 耗时：≈668 分钟
- main / alt pooled OOF：0.69780 / 0.69113（corr≈0.951）
- test Spearman(Plus-max2, best_v1-max2)≈0.983 → 不同文件，但**更弱**
- 产物已落盘：`submissions/submission_super714_plus{,_w62,_wbest}.csv`，`artifacts/super714_plus/`

## 决策

1. **线上继续以 W62（0.71503）为最强提交**，不要用 Plus 系列顶替。  
2. Plus 差异化方向（加深 depth、改分箱、跨世界连续特征）在本数据上伤害信号，记为死路旁证。  
3. 下一步应回到 **best_v1 同构轴** 上找增益：更多 bag/seed、或同臂上已被 W62 验证的加权族微调——而不是再改世界定义。

## 复现融合后处理

```bash
python3 -u src_super/fuse_plus_weights.py
```
