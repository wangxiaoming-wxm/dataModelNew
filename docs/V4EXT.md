# V4-ext（诚实推高阶段最高）

> 提交：`submissions/submission_v4ext.csv`  
> 诚实嵌套（20 block seed）：**0.70382**（sd 见 audit）  
> 对照轨迹：V4 0.70303 → V4∪opus 0.70358 → **+w12 0.70382**  
> 诚实性：`honesty_passed=true`；目标 0.707 / **0.725：均未达成**

## 配方

`views_max_v4_ma_w12`（嵌套多数票）=
`max(rank)` over V4 十臂 ∪ `{merger_ord8, v2_cat_alt8, cat_w12_d5}`  
（`gap_v5` 在含 w12 的规则竞争中常被挤出；`mag_w12` 仍有票。）

| 臂 | bagged OOF | 协议 |
|---|---:|---|
| V4 十臂 | 0.696–0.699 | 固定树，无 ES |
| merger_ord8 / v2_cat_alt8 | 0.6966 / 0.6970 | opus 诚实 |
| cat_w12_d5 | **0.69903**（> main d5 0.69771） | main∪alt 联合 FE，4 seed×10-fold |

## 公开榜校准预估

按你回执 gap ≈0.0092–0.0095：

**预估 LB ≈ 0.7130–0.7133**（期望略高于冠军 0.71222）。

**不能**诚实预估 0.725：Bayes≈0.706 → LB 上限约 0.715–0.717。

## 已拒绝

| 方向 | 结果 |
|---|---|
| mine_noxb_honest | 拖 max |
| w8 fast | 0.68285 < main fast 0.68961 |

## 进行中

- w12 d6l6 袋（与 d5 并列评估）
