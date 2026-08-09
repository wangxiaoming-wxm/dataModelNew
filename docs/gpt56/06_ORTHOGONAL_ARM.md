# S5：经验贝叶斯目标编码与正交低方差臂

## 1. 定位

这是备用增益方向，不是主力替换：

| 臂 | OOF AUC | 与 v2 的 OOF 秩相关（约） | 加入 `max` 后 |
|---|---:|---:|---:|
| `glm` | 0.66489 | 0.895 | 0.69562 |
| `lgb_te` | 0.67110 | 0.923 | 0.69685 |
| v2 `views_max` | 0.69910 | — | — |

当前两个弱臂虽然解耦，但强度差距太大，加入融合反而下降。历史 `plus` 臂提供了一个有用参照：
它约 0.6886、与 v2 相关约 0.939，和 v2 做 max 可到约 0.70049，即有约 +0.0014 的潜在价值。

所以 S5 的硬目标是：

> 新臂单独达到至少 0.685，理想达到 0.688～0.690，同时与 v2 相关 ≤0.93。

达不到 0.685 就停止，不因为“多样性好”把弱臂塞进融合。

## 2. 当前方法的可改进点

`src2/te.py` 对所有类别列使用同一个 smoothing：

```text
lgb_te smoothing = 30
glm    smoothing = 50
```

但列的基数和样本量差异巨大：

- `source` 只有 11 类；
- `region` 20 类；
- `bin_pat` 最多 256 类；
- 三阶交叉可超过 1000 类。

固定 smoothing 会对低基数列过度收缩、对稀疏交叉收缩不足。当前 GLM 也只把所有连续变量放进同一个
SplineTransformer，没有显式表达已证实最强的 `source × condition` 非线性交互。

## 3. E1：目标编码诊断

先不改模型，输出每个 TE 列：

```text
n_categories
median_count
p10_count
singleton_rate
positive_rate_variance
within_category_binomial_variance
estimated_between_category_variance
unseen_rate_in_validation
```

淘汰原则：

- singleton rate >50% 且没有稳定的折外单列 AUC，不编码；
- 折外单列 AUC 在 `[0.49, 0.51]` 且多折不稳定，不进入 GLM；
- `month/version/grades` 不因全量标签率差异看起来大就保留，必须以折外结果为准。

此步骤的目的不是按 OOF 挑几十列，而是删除明显无法估计的高方差统计量。

## 4. E2：经验贝叶斯目标编码

### 4.1 每列估计自己的先验强度

在每个 inner-training fold 内，对某一类别列估计：

```text
p       = inner-training 全局正例率
w_j     = n_j / sum(n_j)
v_obs   = sum_j w_j * (p_j - p)^2
v_noise = sum_j w_j * p*(1-p)/max(n_j, 1)
tau²    = max(v_obs - v_noise, 1e-6)
m       = clip(p*(1-p)/tau² - 1, 2, 500)
alpha   = p*m
beta    = (1-p)*m

posterior_mean = (positive_count + alpha) / (count + alpha + beta)
posterior_var  = ((positive_count+alpha)*(negative_count+beta))
                 / ((count+alpha+beta)^2 * (count+alpha+beta+1))
```

关键约束：

- `m` 只能由 inner-training 行估计；
- outer-validation/test 用 outer-training 的统计量；
- 训练行仍必须 inner cross-fit，任何行不能看到自己的标签；
- 每列独立估计 `m`，并保存每折的估计值；
- 未见类别回退到 `p`，count=0，posterior SD 按先验方差；
- inner splitter seed 固定为 `outer_seed + 1000 + column_index`；
- 同时输出 `log1p(count)` 与 `sqrt(posterior_var)`，帮助模型判断编码可靠性。

### 4.2 对照

只比较三个预登记版本：

```text
fixed_30
fixed_100
empirical_bayes
```

先在 `lgb_te` 上跑，因为单 seed 成本低。若 empirical Bayes 相对 paired `fixed_30`
低于 +0.001，不把它推广到 GLM。

## 5. E3：显式分层样条 GLM

不先引入新的 GAM 依赖，使用现有 scikit-learn 组件构建带 L2 收缩的可解释设计矩阵。

### 5.1 主效应

```text
spline(log_days, 8 knots)
spline(log_cond_r, 8 knots)
spline(log_ratio, 8 knots)
spline(age_range, 5 knots)
8 binary flags + bin_sum
one_hot(source)
one_hot(region)
one_hot(age_cat)
```

### 5.2 只加入有证据的交互

```text
one_hot(source) × spline(log_cond_r)   # 最关键
one_hot(source) × spline(log_ratio)
one_hot(region) × spline(log_days)
one_hot(region × source)               # 需要块级额外收缩
one_hot(bin_pat)                        # 需要块级额外收缩
```

不加入所有样条两两张量积；那会迅速扩大维度并重现盲目交叉过拟合。

### 5.3 正则化

预登记：

```text
LogisticRegression(
  penalty="l2",
  C ∈ {0.01, 0.03, 0.10},
  solver="lbfgs",
  max_iter=5000
)
```

scikit-learn 的单个 `C` 会同等惩罚所有系数，不能天然对高基数块“强收缩”。实现时先标准化其他块，
再将 `region×source`、`bin_pat` 两个 one-hot 块共同乘以预登记缩放 `s ∈ {0.25, 0.5}`；
较小输入尺度会让达到同等效应所需的系数更大，从而承受更强 L2 惩罚。块缩放后不能再次做逐列
标准化，否则收缩会被抵消。

三个 C 与两档块缩放作为有限候选走 discovery/confirmation，不在全量 OOF 上选最优后直接报分。
稀疏矩阵过大时改用支持稀疏输入的 solver，但不改变特征定义。

## 6. E4：低容量 GBDT 作为补充

只有 E2 让 `lgb_te` 提升至少 +0.003 后，才做小范围模型对照：

```text
num_leaves       ∈ {7, 15}
min_child_samples∈ {60, 150}
feature_fraction = 0.6
lambda_l2        = 20
n_estimators     = 450
```

总共 4 个预登记候选。目标不是把 LightGBM 调成主模型，而是把已改善的 TE 组合成一个
0.685+ 且与 CatBoost 解耦的臂。若 E2 后仍低于 0.678，直接停止 E4。

## 7. 验证设计

### 7.1 Discovery

```text
folds = 5
seeds = [20600, 20601, 20602, 20603]
```

因为弱臂成本低，可以严格跑完整配对。每个候选报告：

- 单臂 AUC 和相对当前 GLM/LGB 的差值；
- seed 一致性；
- 与 v2/main/alt 的秩相关；
- `0.1/0.2/0.3` 小权重 rank mean；
- 与 v2 的 rank max，仅作为诊断。

### 7.2 Confirmation

冻结最多 2 个候选，在 `[20610, 20611, 20612, 20613]` 上确认。

进入最终融合必须满足：

| 条件 | 门槛 |
|---|---:|
| 单臂 AUC | ≥0.685；优选 ≥0.688 |
| 与 v2 相关 | ≤0.93 |
| confirmation 融合增益 | ≥+0.001 |
| 正向 seeds | 至少 6/8 |
| bootstrap 正差比例 | ≥90% |

如果单臂达到 0.690 但相关 >0.96，它更像弱版 CatBoost，不是需要的正交臂；只在能替换某臂时保留。

## 8. 实现前置

当前 `te.py` 没有经验贝叶斯接口，也没有分块样条 GLM runner。执行前实现：

```text
src2/te.py
  encode_empirical_bayes(...)

scripts/gpt56_te_diagnostics.py
scripts/gpt56_run_orthogonal.py
scripts/gpt56_orthogonal_report.py
```

所有 empirical-Bayes 超参数必须在对应 inner-training 内估计。脚本接口、输出 schema 和配置
应先提交再运行；上述文件当前不存在，是实现规格。

## 9. 融合规则

正交臂默认不能直接进入 `max`。预登记：

```text
orth_10 = 0.90*v2 + 0.10*orth
orth_20 = 0.80*v2 + 0.20*orth
orth_30 = 0.70*v2 + 0.30*orth
orth_max 仅当 orth_auc ≥ 0.688 时评估
```

权重在 discovery 前写死，由 confirmation 和嵌套外层块决定。即使全量 OOF 显示 0.17 最好，
也不增加 0.17 规则。

## 10. 停止规则与预算

- E2 不能把 LGB 提高至少 +0.001：停止经验贝叶斯在树模型上的扩展；
- 新 GLM <0.678：不做更多张量交互；
- 最佳候选 <0.685：整个 S5 停止；
- 融合增益 <0.001：不进入提交候选；
- S5 最多使用总算力的 10%，主算力仍留给 S2/S3。

合理预期：

- 经验贝叶斯 TE：LGB 单臂 `0～+0.003`；
- 显式 source×condition 样条：GLM 单臂 `+0.005～+0.015`，但从 0.665 到 0.685 仍是高难度；
- 若达到门槛，最终融合约 `+0.001～+0.003`。

这是高不确定性方向，达到停止门槛后应果断结束。

## 11. 完成定义

- 每列 TE 的可靠性诊断齐全；
- 经验贝叶斯超参数严格由训练折估计；
- GLM 只含有数据证据支持的交互；
- discovery/confirmation 分离；
- seed 分离只检验随机稳定性；最终统计声明需执行 S6 的最外层按行嵌套开发，
  否则标为给定候选后的条件评估；
- 达不到 0.685 明确停止；
- 通过时只作为 10%～30% 的正交臂进入冻结融合。

