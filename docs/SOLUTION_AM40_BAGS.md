# 超越 W62：AM40 + SUPER714-Bags

目标：最终提交的本地 OOF **严格大于** 冻结 W62（0.70159366）。线上 W62 = 0.71503。

## 1) AM40（秒级，已通过门禁）

在冻结 best_v1 双臂上：

```text
linear = 0.62 * main + 0.38 * alt
score  = 0.40 * max(main, alt) + 0.60 * linear
```

| 指标 | 值 |
|---|---|
| AM40 OOF | **0.70181135** |
| W62 OOF | 0.70159366 |
| Δ | **+0.00021769** |
| Spearman vs W62 提交 | ≈0.99893（不同文件） |

```bash
bash run_am40.sh
bash run_am40.sh --verify
```

主文件：`submissions/submission_am40.csv`（当前亦复制为 `submission_beat_w62.csv`）。

## 2) SUPER714-Bags（训练中）

同构 best_v1，**只把 bags 从 3 提到 6**（种子/深度/分箱/损失不变），再比较 max2 / W62 / AM40 / wbest。  
硬门禁：最优融合 OOF > 冻结 W62 才晋升 `submission_beat_w62.csv`。

```bash
bash run_super714_bags.sh --smoke
bash run_super714_bags.sh
```

日志：`logs/super714_bags_full.log`

## 决策规则

1. 始终持有一个 `submission_beat_w62.csv`，其 OOF 必须 > W62  
2. 当前地板 = **AM40**  
3. Bags 若再高，覆盖 beat 文件；否则保留 AM40  
