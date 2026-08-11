# SUPER714-Extend（best_v1 同构 + 新 seed）

在 **不改特征/超参/分箱** 的前提下，把冻结 best_v1 的 8 个 seed 与新 seed `2034–2037` 做 rank-pool 加权合并，再输出 max2 / W62 / OOF-wbest。

## 为何这条线

| 实验 | 结论 |
|---|---|
| Plus（改 depth/分箱/跨世界） | OOF 0.696，**拒绝** |
| W62（冻结臂加权） | 本地 0.70159，线上 **0.71503** |
| Extend | 只加 seed，保留 W62 融合 |

合并公式：

```text
merged = (8 * frozen_pool + 4 * new_pool) / 12
```

其中 `new_pool` 为新 seed 的 rank 均值（协议与 best_v1 完全一致：main Ordered d5×800×l2=10；alt Plain d6×800×l2=6；3 bags×5 folds）。

## 复现

```bash
bash run_super714_extend.sh --smoke
bash run_super714_extend.sh
```

产物：

- `submissions/submission_super714_extend{,_w62,_wbest}.csv`
- `artifacts/super714_extend/metrics.json`
