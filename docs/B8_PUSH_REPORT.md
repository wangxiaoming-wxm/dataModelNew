# B8 冲分进展（诚实口径）

> 相对 B7 closest nested **0.702704955**；门禁仍为 **0.71**。

## 当前权威交付

| 口径 | 值 |
|---|---:|
| B7 closest max3 | 0.702704955 |
| **B8 segment-gate nested** | **0.703374**（菜单嵌套；多数规则 `s_M_v10_22_age6` full=0.703485） |
| Δ vs B7 | **+0.000780** |
| gate 0.71 | FAIL |

提交：`submissions/submission_b8_closest_honest.csv`  
复算：`python3 scripts/b8_segment_gate.py`

### 预注册分段规则（主报）

在冻结三臂 `gap / gap_bag / plus` 上：

1. 若 `grades==s` 或 `t3_sfx==M` 或 `version==v10` 或 `region==22b5` → 用 `b6max=max(gap,gap_bag)`
2. 否则若 `age_range==6` → 用 `plus`
3. 否则 → `max3`

嵌套菜单选规则（含 max3 / gate_s / gate_sm / 多个复合门）后，5 折多数选中上述复合规则；相对 max3 的 2-way 嵌套在多个外层 seed 上 5/5 稳定。

## 已试且未超过该门控的路径

| 路径 | 结果摘要 |
|---|---|
| 扩展离散三臂融合（power/median/rank…） | 仍选 max3，无增益 |
| meta stack（logit/HGB/RF on OOF） | solo≤0.699；max 后 ≤0.70312 且不稳 |
| gapv3 新交叉特征 CatBoost | solo≈0.694；max4 伤分 |
| gapv3 lossguide / baghot / LGB | 更弱 |
| embedding MLP | solo≈0.60，伤分 |
| YetiRank（region group） | solo≈0.645；伤分 |
| 复用 b6pro keepx/nodays/aging/baghot | max4 ≤ B7；高相关同质 |
| **全数据贪心分段补丁（可抬到 ~0.7057）** | **诚实嵌套发现仅 ~0.7022；内层校验回退到 0.703485** → 过拟合，禁止交付 |
| plus 5fold 变体屏（deep/wide） | 5fold 低估；未超门控地板 |

历史 b6pro 本地 0.71 链已因公开榜 0.70208 作废，**不引用**。

## 关键教训（B8）

- 分段门控对「max 伤害切片」有效，但是小幅（+0.0007）
- 在全量 OOF 上堆叠许多切片规则，本地可虚高到 0.705+，**必须**用折外发现/内层校验；一经诚实嵌套即塌
- 同质 CatBoost 微扰 / 弱异构臂无法通过 `max` 抬分

## 下一步（继续冲 0.71）

1. 找 **solo≳0.69 且 corr(B6)≲0.90** 的新异构臂（不同损失/表模型/表征）
2. 在分段门控地板上做 max(门控, 新臂)，并保持嵌套预注册
3. 禁止单 seed ultra patch / 连续 α 搜权 / 全量贪心切片冒充 nested
