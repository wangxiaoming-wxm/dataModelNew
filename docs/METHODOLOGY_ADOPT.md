# 方法论接入（beat-max3）

按 search-first：先采用现成工具，再蒸馏成可复现臂。

## 已采用

| 来源 | 用法 | 落点 |
|---|---|---|
| **features_goldmine** (PyPI) | fold-local `GoldenFeatures.fit_transform`；排除 `categorical_oof_target` | `src_beat/train_method_arm.py --mode goldmine` |
| **CoFEH 思想** | ToT 式算子探索 → 蒸馏确定性 FE（age×days、source 组偏差、days/cond 规则） | `--mode cofeh` |
| **Made-With-ML** | 第一性原理：残差分析显示 plus 在低分段互补 → 新臂保持异构（LightGBM+新 FE） | 训练设计 |
| **verification-loop / 监督门禁** | 新臂必须过 `supervise.py`（Δ/Spearman/blocks+/保留 noxb） | `src_beat/supervise.py` |
| **SkillHub** | 尝试安装 ML/FE 技能；部分 registry 缺 `SKILL.md` 失败，改用 PyPI+蒸馏 | 见下 |

## SkillHub 状态

- `npx skillhub search` 可用；若干热门技能仓库路径无 `SKILL.md`，`--no-api` 安装失败。
- 不阻塞：已用 `features-goldmine` + `featuretools`（备选）落地。

## 队列

`run_beat_max3_followup.sh` 在 P1–P3 后跑：

1. plus_new8  
2. cofeh_arm（8 seed）  
3. goldmine_arm（4 seed）  
4. 全部过监督门禁后更新 leaderboard  

## 硬约束不变

- 底座三臂不删；融合 `max(rank)`  
- 禁止 nested+0.0095 外推 LB  
- 首选交榜仍为 `submission_max3_best.csv`，直到有更高门禁候选
