# V4-ext（诚实推高阶段最高）

> 提交：`submissions/submission_v4ext.csv`  
> 诚实嵌套：**0.70381–0.70382**（8-seed w12 与 4-seed 嵌套持平）  
> 轨迹：0.70303 → 0.70358 → **0.70382**  
> `honesty_passed=true`；**0.725 不能诚实承诺**

## 配方

`views_max_v4_ma_w12` = max(rank) over V4十臂 ∪ merger_ord8 ∪ v2_cat_alt8 ∪ cat_w12_d5(8seed)

| 臂 | bagged OOF |
|---|---:|
| cat_w12_d5 (8 seed) | **0.69986** |
| merger_ord8 / v2_cat_alt8 | 0.6966 / 0.6970 |

## 预估公开榜

**≈ 0.7130–0.7133**（gap 0.0092–0.0095）。现冠军 0.71222。

Bayes in-sample isotonic（当前融合）≈ **0.7075** → 诚实 LB 上限约 **0.715–0.717**。  
距 0.725 仍差 ≈0.008+，**不是继续堆种子能补的缺口**。

## 已拒绝

mine_noxb_honest；w8；w12 d6l6（nested 下降）；8-seed 相对 4-seed **无嵌套增益**（仅降噪）。

## 进行中

Ordered 固定树筛 alt / w12 / main。

## 续拒（2026-08-10 晚）

| 方向 | 同种子对照 | 裁决 |
|---|---|---|
| Ordered×alt | 0.68738 < Plain 0.68843 | 拒 |
| Ordered×w12 | 0.69247 < Plain 0.69398 | 拒 |
| Ordered×main | 0.69296 > Plain 0.68895 | 已有 merger_ord8，不重复 |
| PairLogit×source/reg_src | OOF ≈0.62–0.65 | 拒 |
| RMSE 固定树×main/alt | bag ≈0.690 << cls 0.698 | 拒 |

边际收益已进入噪声带（最近一步嵌套 Δ≈0）。
