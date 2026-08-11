# Beat max3 — 当前可交最高版

## 提交文件（主推）

**`submissions/submission_beat_max3.csv`**

| 项 | 值 |
|---|---|
| 配方 | `max(rank(merger_ord8), rank(v2_cat_alt8), rank(ord_noxb_strong), rank(plus_strong))` |
| `ord_noxb_strong` | `0.5·ord_noxb_bag + 0.5·new16_bag`（同一 ES 逻辑臂加种子，非孪生堆叠） |
| nested Δ vs v4_max3 | **+0.0016 ~ +0.0017**（随 new16 种子数浮动） |
| Spearman vs v4_max3 | ≈ **0.9915** |
| blocks+ | **4/5** |
| 臂数 | **4**（策略合规） |

对照冠军：`submissions/submission_v4_max3.csv`（LB **0.71222**）。

## 明确不交

- `submission_max3_best*` / `*_pro*` / kitchen-sink（noxb10+w12）— v4ext 同款风险
- `plus+b7` 双互补 max 堆叠 — 二者 Spearman≈**0.977**，属高相关孪生

## 备选（同门禁，非主推）

- `submission_ship_max3s_b7.csv`：第四臂换 `b7_closest`（其本身已是多源 fuse，迁移风险更高）
- `submission_ship_max3_plus.csv`：未加强 ord 的四臂版

## 诚实边界

本地 nested 增益**不能保证**公开榜 > 0.71222。本版是策略约束下期望值最高的可交候选；探针正交臂（exp1/2/3）若过门禁会自动替换进 `submission_beat_max3.csv`。
