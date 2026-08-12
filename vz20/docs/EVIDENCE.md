# vz20 换轴证据（诚实，未过夺冠门禁）

门禁（预注册）：相对冻结 W62 OOF 0.70159，新臂融合需 **+0.001** 且 test Spearman(W62) **< 0.995**（避免再交同构票）。

## 已验证的线上事实

- W62 LB **0.71503**（OOF 0.70159，gap +0.01344）——**用户已交，禁止再交**
- vz17 LB 0.71487
- vz19 LB **0.71298**（OOF 0.70355，gap 仅 +0.0094）→ 本地「改进」伤害了线上
- AM40 与 W62 test Spearman≈0.999，**同样不该交**

## 本轮独立实验（均未晋级）

### A. 先前轮次

| 方向 | 协议 | 结果 | 裁决 |
|---|---|---|---|
| rebuild V2 诚实 nested | 5×3×2 | 0.69518 | 低于 W62，淘汰 |
| 原生 LightGBM（原始类别列） | 3-fold | 0.635，融 W62 下降 | 淘汰 |
| 原始列 CatBoost RMSE | 3-fold | 0.659，融 W62 下降 | 淘汰 |
| id byte 作为 Ordered 类别 | 3-fold | 略低于无 id 基线 | 淘汰 |
| x0–x18 向量（logit/PCA/ET） | 5-fold | 0.51–0.52 | 无联合强信号 |
| SUPER714 bags 3→6 | 全量 | AM40 0.70160 < 0.70181 | 更多 bag 无增益 |

### B. 本轮：在 `build_main` 世界上换损失 / 几何 / 切片（3-fold × 400 iter）

同预算 Plain RMSE 对照 **0.68680**。弱于冻结 W62 是预算差，不是实现 bug。

| 方向 | 臂 OOF | 融 W62 最优权重 | 裁决 |
|---|---:|---:|---|
| CatBoostClassifier Logloss | 0.68686 | 0 | 与 RMSE 同构，先前「Logloss≈0.51」是错误实现 |
| Logloss + Balanced | 0.68105 | 0 | 淘汰 |
| 逆频率加权 RMSE | 0.68187 | 0 | 淘汰 |
| LightGBM binary **on build_main** | 0.63828 | 0 | 类别处理远弱于 CatBoost |
| YetiRank（source 分组，已排序） | 0.64179 | 0 | 弱于 RMSE |
| PairLogit / 单查询 YetiRank | 失败或过慢 | — | 淘汰 |
| f09d 切片专家再拼接 | 切片 0.633 < W62 切片 0.659 | 全局 -0.006 | 少样本专家弱于全局 |
| CAR_2 切片专家 | 切片 0.615 < 0.667 | 全局 -0.011 | 同上 |
| KNN（log_days+cond / source 内 / main 数值） | 0.596–0.624 | 0 | 淘汰 |
| source×age slim 第三世界 | 0.64910 | 0 | 瘦特征不够 |
| Langevin / Bernoulli / rsm0.8 / Huber / Quantile0.9 | ≤0.685 | 0 | 淘汰 |
| score_function=L2 | 0.68447 | 0 | 弱于默认 Cosine |

### C. 融合几何（零训练，已穷尽）

| 配方 | OOF | vs W62 |
|---|---:|---:|
| W62 0.62/0.38 | 0.70159 | — |
| 线性最优 0.59/0.41 | 0.70160 | +0.000006 |
| AM40 | 0.70181 | +0.00022，Spearman≈0.999 |
| logit 空间混合 | 0.70182 | +0.00022，同构 |
| source 内 rank / source-z | ≤W62 | 权重 0 |
| ratio rank 混入 | 权重 0 | 已被树吃掉 |

### D. 数据切片（解释天花板，不是新票）

- 大区 `f09d`（n=3560，24%）W62 AUC **0.659** vs 其余 **0.709**
- `CAR_2|ENG_262`（n=3586）W62 AUC **0.667** vs 其余 **0.712**
- 难正例（最低 ratio 的 200 个正例）集中在 month=M2、grades=ss；单列交互没有稳定 TE
- `x0..x18` 残差 logit 叠 W62：**0.699 < 0.702**（加噪声）
- livability 不是 region 的 1-1；t3 解析成数字+字母无单体信号
- train/test 无 id 重叠、无整行重复；id 整数/行号无泄漏

## 仍在跑：Ordered 内部超参

3-fold×400 上 `fold_len_multiplier` 1.5/3.0 相对默认 Ordered 有 **+0.001～0.0015** 苗头（可能是噪声）。  
正在用 **5-fold × 800 iter** 确认 `fold_len_multiplier` 与 `fold_permutation_block`，并测新交叉 / f09d 上采样。  
只有相对**同预算默认 Ordered** 稳定 ≥ +0.001 才升格为全预算新主臂。

## 冻结对照

| 配方 | OOF |
|---|---:|
| best_v1 main | 0.69992 |
| best_v1 alt | 0.69770 |
| W62 | **0.70159** |
| max2 | 0.70128 |
| AM40 | 0.70181 |

## 对 0.72 / 0.749 的含义

冠军相对 W62 约 **+0.005 / +0.034**。当前数据上：

- 被忽略的 `x0..x18` 在残差里仍是噪声；
- 分类损失、排序损失、KNN、切片专家、Langevin 都不能在 `build_main` 上接近 RMSE；
- 再磨 main/alt 融合权重只有 0.0002 级同构差异。

**在 Ordered 内部超参确认之前，没有可交的新排名。禁止再交 W62 / AM40。**
