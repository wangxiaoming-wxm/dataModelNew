# 特征工程设计文档 — best v1（线上 0.71464）

> 对应代码：`explore_best.py`（v1 版本）
> 本文档整理 best v1 完整特征体系，分析其效果优秀的原因。

---

## 0. 总体架构

```
原始数据 (45列)
  ├─ 编码世界1 (cond_r世界): condition按source归一化 → 臂1 (Ordered+d5+RMSE)
  ├─ 编码世界2 (rate世界):   condition按source排名 → 臂2 (Plain+d6+RMSE)
  └─ max2融合: 逐元素 max(rank_oof) → 本地 0.70128, 线上 0.71464
```

**核心设计哲学**：同一份信息用两种不同的"编码语言"表达，让两个模型学到互补的排序信号，再通过 max 融合取各自高置信的预测。

---

## 1. 编码世界1：cond_r 世界（臂1，0.69992）

### 1.1 核心创新：condition 按 source 归一化

**问题**：不同车型（source）的 condition 尺度差异极大。CAR_1 的 condition 均值 0.30，CAR_2 均值 0.92，相差 3 倍。直接用原始 condition 做特征，模型需要额外学习"这个 condition 值属于哪个车型"的上下文。

**方案**：
```
cond_r = condition / median_condition_of_source
```

**效果**：`ratio = days / cond_r` 单特征 AUC = 0.620，比 `days` 原始 AUC=0.593 高 **+0.027**。这是整个方案中**最强的单特征**。

### 1.2 数值特征（~30个）

| 类别 | 特征 | 含义 | 关键特征标记 |
|---|---|---|---|
| **时间** | days, log_days | 距首次投保天数 | days AUC=0.593 |
| **车况** | condition, log_condition, condition_missing | 原始车况 + 缺失标志 | condition_missing 本身是强信号 |
| **归一化车况** | cond_r, log_cond_r | condition/median(source) | 消除车型间尺度差异 |
| **核心比值** | ratio, log_ratio, ratio_p75 | days/cond_r, days/cond_r^0.75 | **ratio AUC=0.620，最强单特征** |
| **交互** | cond_x_days, cond_over_days | condition×days, condition/days | 捕获时间×车况的非线性关系 |
| **年龄** | age_range, days_over_age | 年龄区间, 时长/年龄 | |
| **等级** | grade_ord | grades(s/ss/sss)转数值(1/2/3) | |
| **标志位** | t1,t2,r1,r2,c1,c2,w1,w2, bin_sum | 8个二值标志 + 求和 | bin_sum 作为聚合信号 |

### 1.3 类别特征与交叉（~80个）

#### 基础类别（8个）
`region, source, month, version, grades_c, age_cat, bin_pat, days_fx`

**bin_pat** = 8个标志位拼接（如 "01001101"），170种组合。这是 opus 作者通过 EDA 发现的数据结构——8个 flag 的组合模式比单独使用更有区分力。

**days_fx** = days 的固定分桶（[700, 2500, 5000, 7000, 9000, 10000]），7个桶。用业务语义边界而非等频分桶。

#### 多尺度分桶（每列4档：5/10/20/40分位）
```
days_q{5,10,20,40}   — days 的 4 种粗细分桶
ratio_q{5,10,20,40}  — ratio 的 4 种粗细分桶（核心！）
cond_q{5,10,20}      — condition 的 3 种分桶
condr_q{5,10,20}     — cond_r 的 3 种分桶
```

**设计意图**：不同粒度的分桶让 CatBoost 在不同尺度上学习交互。5分桶捕获粗粒度模式（如"高ratio vs 低ratio"），40分桶捕获细粒度差异。

#### 二阶交叉（精选）
| 交叉名 | 组成 | 业务含义 |
|---|---|---|
| d10c10 | days_q10 × cond_q10 | 时间×车况 联合网格 |
| d10r, d20r | days分桶 × region | 各地区的时长分布 |
| d10s, d20s | days分桶 × source | 各车型的时长分布 |
| c10r, c10s | cond分桶 × region/source | 各地区/车型的车况分布 |
| r10r, r10s, r10a | ratio分桶 × region/source/age | **ratio 与各维度的交互** |
| r20r | ratio_q20 × region | 细粒度ratio×地区 |
| cr10r, cr10a | condr分桶 × region/age | 归一化车况的交互 |
| d10p, r10p, rp | days/ratio × bin_pat | flag组合模式交互 |
| ra, sa | region/source × age_cat | 地区/车型 × 年龄 |

#### 三阶交叉
| 交叉名 | 组成 |
|---|---|
| d5rs, r5rs | days_q5/ratio_q5 × region × source |
| d10c10r, d10c10s, d10c10a | days_q10 × cond_q10 × region/source/age |
| sc10a, rc10a | source/region × cond_q10 × age_cat |
| rsa | region × source × age_cat |
| dfs, dfc10, dfcr10, dfr | days_fx × source/cond/condr/region |

#### 条件×source 的深度交叉（level2）
`condition × source` 是最强的交互方向。此处用多个独立离散化来捕获：
```
c5s, c20s        — cond_q5/q20 × source
cr5s, cr10s, cr20s — condr_q5/q10/q20 × source  
cr5r, cr20r, c5r  — condr/cond × region
d5c5, d20c20      — days × cond 联合分桶
d5cr5, d10cr10    — days × condr 联合分桶
```

### 1.4 频率编码（label-free）
对以下列做 count encoding（不碰标签，零泄漏）：
```
region, source, bin_pat, rs(region×source), 
d10r(days_q10×region), c10s(cond_q10×source),
month, version
```

### 1.5 噪声视图（x19/x20/livability/t3/code 作为类别+交叉）
opus 作者通过 EDA 发现这些列是 source/condition/region 的带噪副本。它们本身无独立信号，但作为"不同扰动的离散化"可以起到隐式 bagging 效果：

| 特征 | 本质 | 交叉 |
|---|---|---|
| x19_cat | V 的 16 级评分量化 | x19_cat × liv_cat |
| x20_cat | condition 相关评分（125个值） | x20_cat × source/region/age |
| liv_cat | livability 类别 | region × liv_cat, liv_cat × age |
| t3_cat | 档次类别 | t3_cat × days_q5 |
| code_cat | 车辆级别 | — |

以及 `cc, max_g, V` 作为数值直通。

### 1.6 训练配置
| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | CatBoostRegressor (RMSE) | D方案验证 RMSE > Logloss |
| boosting | **Ordered** | 类别特征多时更稳定 |
| depth | 5 | 防过拟合 |
| iterations | 800 | 固定树数，无早停（诚实 OOF） |
| l2_leaf_reg | 10 | L2 正则 |
| seed | 8个 (2026~2033) | 多视角 rank-pool |
| bagging | 3个内种子 | D方案验证的稳定性增益 |
| CV | 5-fold StratifiedKFold | 分层抽样 |

---

## 2. 编码世界2：rate 世界（臂2，0.69770）

### 2.1 核心创新：condition 按 source 排名

**与臂1的区别**：臂1 用 `condition / median`（比例归一化），臂2 用 `rank_pct`（排序归一化）。

```
rk = rank_pct(condition within source)
rate = days × (1 - rk)     ← 核心特征
```

`rate` 单特征 AUC = 0.622，略高于 `ratio=0.620`。

**设计意图**：rank 归一化不受 condition 分布的偏度影响（CAR_1 的 condition 偏左，median 归一化可能不稳定），提供另一种"条件×时间"的表达方式。

### 2.2 数值特征（~15个）
```
days, sqrt_days, condition, cond_rk, rate, log_rate, 
rate_over_age, condition_missing, age_range, grade_ord,
t1~w2 (8个flag), bin_sum
```

相比臂1更精简，减少了 cond_r/ratio 系列的派生特征，增加了 sqrt_days（开根变换）和 rate_over_age。

### 2.3 类别特征与交叉（~35个）

**基础类别**：`region, source, month, version, grades_c, age_cat, bin_pat`

**多尺度分桶**（3档：7/13/25分位）：
```
d{7,13,25}  — days 分桶
k{7,13,25}  — cond_rk 分桶（rank分位）
e{7,13,25}  — rate 分桶
```

**精选交叉**（比臂1更聚焦 condition×source）：
| 交叉 | 含义 |
|---|---|
| Ak7s, Ak13s, Ak25s | cond_rk × source（3个尺度） |
| Ak13r, Ak7a | cond_rk × region/age |
| Ad13r, Ad13s, Ad7a, Ad25r | days分桶 × region/source/age |
| Ae13r, Ae13s, Ae7a, Ae7p | rate分桶 × region/source/age/pat |
| Ad7k7, Ad13k13 | days × cond_rk 联合分桶 |
| Ars, Ara, Asa | region×source, region×age, source×age |
| Ad7rs, Ak7ra, Ae7rs | 三阶：days/cond_rk/rate × region×source/age |

### 2.4 训练配置（与臂1的差异）
| 参数 | 臂1 | 臂2 | 差异原因 |
|---|---|---|---|
| boosting | Ordered | **Plain** | 不同算法增加多样性 |
| depth | 5 | **6** | 更深树捕获不同模式 |
| l2_leaf_reg | 10 | **6** | 更弱正则化 |
| iterations | 800 | 800 | 相同 |

**关键设计**：两臂用不同的 boosting 类型 + 不同的 depth + 不同的 l2 → 降低相关性（0.948），让 max 融合有增益空间。

---

## 3. 融合策略：max2(rank)

```
臂1 OOF → rank归一化 → [0,1]
臂2 OOF → rank归一化 → [0,1]
max2 = max(臂1_rank, 臂2_rank)
```

**为什么用 max 而非 mean**：
- 臂1~臂2 相关性 = 0.948（高度相关但非完全相同）
- max 对"哪些样本被顶到高位"更敏感
- max 增益 ≈ (1 - 0.948) × 0.698 ≈ 0.036 → 本地融合增益 = 0.70128 - 0.69992 = 0.00136

**为什么用 rank 而非原始概率**：
- rank 归一化消除两臂预测尺度的差异
- 让 max 操作在统一的 [0,1] 空间中进行

---

## 4. 效果优秀的原因分析

### 4.1 特征层面

| 排名 | 因素 | 贡献 | 证据 |
|---|---|---|---|
| 1 | **condition 按 source 归一化** | 核心突破 | ratio AUC=0.620 vs days=0.593, +0.027 |
| 2 | **ratio 多尺度分桶 + 交叉** | 结构红利 | ratio_q5/q10/q20/q40 × region/source/age 等 |
| 3 | **condition × source 深度交叉** | 最强交互 | c5s/cr5s/cr10s 等多尺度版本 |
| 4 | **x19/x20 作为类别交叉** | 隐式 bagging | 利用噪声编码的多样性 |
| 5 | **bin_pat 组合模式** | 结构化 | 8个flag的170种组合，比单独使用更有区分力 |
| 6 | **频率编码** | 先验信息 | count encoding 提供类别基数信息 |

### 4.2 训练层面

| 排名 | 因素 | 贡献 |
|---|---|---|
| 1 | **RMSE 替代 Logloss** | D方案验证 +0.00135 |
| 2 | **Ordered vs Plain 双 boosting** | 降低臂间相关性 |
| 3 | **depth5 vs depth6** | 不同复杂度捕获不同模式 |
| 4 | **固定 800 iter 无早停** | 诚实 OOF，无乐观偏差 |
| 5 | **8 seed rank-pool** | 多视角平均 |
| 6 | **3 bagging** | D方案验证的稳定性增益 |

### 4.3 融合层面

| 排名 | 因素 | 贡献 |
|---|---|---|
| 1 | **双编码世界** | 同信息不同表达 → 互补信号 |
| 2 | **max 融合** | 对高置信样本敏感 |
| 3 | **rank 归一化** | 消除尺度差异 |

### 4.4 为什么比 D方案（0.70457）好 +0.01047

| 差异维度 | D方案 | best v1 | 增量估算 |
|---|---|---|---|
| condition归一化 | 无（原始 condition） | cond_r = condition/median(source) | +0.005~0.008 |
| ratio=days/cond_r | 无 | 单特征 AUC=0.620 | 核心增量 |
| 编码世界 | 单一（v33特征） | 双编码世界（cond_r + rate） | +0.002~0.003 |
| 训练策略 | Plain+d6+ES | Ordered+d5 vs Plain+d6 | +0.001~0.002 |
| 融合 | 无 | max2(rank) | +0.00136 |

### 4.5 为什么比 opus5 原版（0.711）好 +0.00364

| 差异 | opus5 原版 | best v1 |
|---|---|---|
| 损失函数 | Logloss | **RMSE** |
| bagging | 无 | **3 bagging** |
| 效果 | 本地 0.69993 | 本地 **0.70128** |

RMSE + 3bagging 在 opus5 特征基座上额外贡献了 +0.00135。

---

## 5. 特征血缘图

```
原始数据
├─ 编码世界1: cond_r = condition / median(source)
│   ├─ 数值: days, condition, cond_r, ratio, log_ratio, ratio_p75, 
│   │        cond_x_days, cond_over_days, age_range, days_over_age,
│   │        grade_ord, t1~w2, bin_sum
│   ├─ 分桶: days_q{5,10,20,40}, ratio_q{5,10,20,40}, 
│   │        cond_q{5,10,20}, condr_q{5,10,20}
│   ├─ 交叉: d10c10, d10r/s, c10r/s, r10r/s/a, cr10r/a,
│   │        c5s/cr5s/cr10s, d5c5/d5cr5, d10c10r/s/a, rsa, r5rs, ...
│   ├─ 频率: freq_{region,source,bin_pat,rs,d10r,c10s,month,version}
│   ├─ 噪声: x19c/x20c/lvc/t3c/cdc + 交叉(x20s/r/a, x19l, lva, rl, ...)
│   └─ 直通: cc, max_g, V
│       │
│       ▼ CatBoostRegressor(Ordered, depth=5, iter=800, 8seed×3bag)
│       │ OOF = 0.69992
│
├─ 编码世界2: rk = rank_pct(condition | source)
│   ├─ 数值: days, sqrt_days, condition, cond_rk, rate, log_rate,
│   │        rate_over_age, condition_missing, age_range, grade_ord, 
│   │        t1~w2, bin_sum
│   ├─ 分桶: d{7,13,25}, k{7,13,25}, e{7,13,25}
│   ├─ 交叉: Ak7s/k13s/k25s, Ak13r/k7a, Ad13r/s, Ae13r/s, 
│   │        Ad7k7/Ad13k13, Ars/Ara/Asa, Ad7rs/Ak7ra/Ae7rs, ...
│   ├─ 频率: freq_{region,source,bin_pat,Ars,Ak13s,Ad13r}
│   └─ 噪声: x19c/x20c/lvc/t3c/cdc + 交叉
│       │
│       ▼ CatBoostRegressor(Plain, depth=6, iter=800, 8seed×3bag, l2=6)
│       │ OOF = 0.69770
│
└─ max2(rank_oof_arm1, rank_oof_arm2)
    │ OOF = 0.70128
    │ 线上 = 0.71464
```

---

## 6. 关键设计决策与消融证据

| 决策 | 验证 | 结论 |
|---|---|---|
| RMSE vs Logloss | D方案实验 | RMSE +0.00135 |
| Ordered+d5 vs Plain+d6 | v3实验 | Ordered让臂2变强但相关性也升，Plain+d6恰好平衡 |
| 3bagging vs 无bagging | vs opus5原版 | +0.00135 |
| 8seed vs 12seed | v2实验 | 12seed反降臂2(-0.00057) |
| 第3编码世界 | v2/v5实验 | 臂3太弱(0.695/0.694)，max3反降 |
| 新比值特征(V/cond_r等) | v6实验 | 待验证 |
| v33特征叠加 | 消融实验 | 叠加无益 |

---

*文档生成：2026-08-11，对应版本 explore_best.py v1（线上 0.71464）*
