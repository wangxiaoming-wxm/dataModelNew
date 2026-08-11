# Beat max3 — 接入 714 best_v1

## 参考来源
GitHub `main/714.zip` → `refs_incoming/714/`
- `特征设计文档_best_v1.md`（声称线上 **0.71464**）
- `explore_best.py`（双编码世界 max2）

## 配方
- 臂1：cond_r 世界 + RMSE + Ordered d5 iter800 + 8seed×3bag
- 臂2：rate 世界 + RMSE + Plain d6 iter800 + 8seed×3bag
- 融合：`max(rank(main), rank(alt))`

## 相对 v4_max3（LB 0.71222）
若复现成功，best_v1 是当前**期望值最高**的可交候选（文档声称 +0.002+ LB）。

## 当前主推（训练完成前）
仍为 `submissions/submission_beat_max3.csv` = `ship_max3s_plus`（真·4臂，Δ≈+0.0018 nested）。

训练完成后：
- `submissions/submission_best_v1.csv`
- 若过门禁，自动提升为 `submission_beat_max3.csv`
