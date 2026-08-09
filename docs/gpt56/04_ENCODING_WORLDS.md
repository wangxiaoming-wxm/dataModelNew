# S3：强编码世界工厂

## 1. 为什么这是主攻方向

现有消融中，主要增益来自“同一有效信息的不同编码”：

| 改动 | 结果 |
|---|---:|
| 干净特征 + 手工交叉 | bag OOF 0.68225 |
| 加主办方噪声编码副本 | 0.69209 |
| 加 4 套可控 jitter | 0.69386 |
| main + alt 多世界融合 | 0.69910 |

相比之下，改 CatBoost 深度的两臂相关约 0.998，几乎不产生互补；继续加普通种子也已饱和。
因此 S3 的目标不是制造更多任意特征，而是：

> 保住 `cond_r` / `ratio` 的强信号，用统计上合理的不同方式离散化和交叉，
> 得到与 main 同等强、但秩相关更低的 CatBoost 臂。

## 2. 先修复 alt2，而不是直接随机搜索

`cat_alt2` 只有 0.69100，比 main/alt 低约 0.004。代码中有三个可解释原因：

1. `build_alt2` 用 `region × source` 小组的均值和标准差，格子估计方差大；
2. `load = cz - 2*dpc` 的系数 2.0 没有实验依据；
3. alt2 没有保留 main 中已证实最强的 `cond_r`、`ratio` 和 `log_ratio`。

第三点最关键：新编码世界应该改变表达，不应该丢掉已知强信号。

### 2.1 alt2-repair 冻结设计

在 alt2 原有列之外无条件加回：

```text
cond_r
log_cond_r
ratio
log_ratio
ratio_p75
```

将小组统计改为无标签收缩：

```text
w = n_group / (n_group + 50)
center = w * median(condition | region, source)
       + (1-w) * median(condition | source)

scale = w * IQR(condition | region, source)
      + (1-w) * IQR(condition | source)

cond_robust = (condition - center) / max(scale, eps)
```

先固定 `k=50`，不在同一轮扫 10/20/50/100。删除任意系数 `2.0` 的 `load`，改用独立的
`cond_robust` 和 `days_percentile` 分箱，让 CatBoost 自己组合。

## 3. 四个有明确假设的候选世界

所有候选都保留：

- main 的核心数值：`days`、`cond_r`、`ratio`、`age_range`、8 个二值开关；
- `region`、`source`、`age_cat`、`bin_pat`；
- 主办方噪声视图；
- 3～4 套确定性 jitter；
- 20～35 个精选交叉的上限。

### W1：robust-source

目的：用中位数/MAD 替代均值/标准差，降低 condition 长尾和异常值影响。

```text
z_robust = (condition - median(condition | source))
           / (1.4826 * MAD(condition | source) + eps)
```

分箱：`z_robust ∈ {6, 12, 24}`；`days/ratio ∈ {8, 16, 32}`。
交叉只保留 `z × source/region/age`、`ratio × region/source` 和已证实的 segment 交叉。

### W2：shrunk-region-source

目的：利用 region×source 的局部结构，同时通过 W1 的收缩公式避免 alt2 小格子过拟合。

额外生成：

```text
cond_shrunk
rank(condition | region, source) 的样本量收缩版本
days_percentile_within_region
```

只与 `source`、`region`、`age_cat` 做精选交叉，不重新做全配对。

### W3：quantile-normal

目的：把 source 内 condition 百分位映射成近似高斯分数，让等宽切分与 alt 的等频切分产生不同边界。

```text
u = clipped_rank(condition | source)
qnorm = Φ⁻¹(u)
```

同时保留原 `ratio`。分箱使用等宽 `qnorm` 箱和 `log_ratio` 等宽箱，而不是再次使用全局分位箱。
该世界的价值是不同边界，不是新增信息。

### W4：offset-binning

目的：对同一连续信号使用错位切点，降低某一组分位边界的偶然性。

对 `days`、`cond_r`、`ratio` 各建立两套相同箱数但半箱错位的切分：

```text
base quantiles   = 0.10, 0.20, ..., 0.90
offset quantiles = 0.05, 0.15, ..., 0.95
```

只把两套编码分配到不同模型，不在一个模型里堆满全部列，避免重现“交叉越多越差”。

## 4. 参数空间必须受限

第一轮最多 8～12 个配置，参数只从下列预登记集合取值：

```text
normalization ∈ {alt2_repair, robust_source, shrunk_regsrc, qnorm}
bin_scheme    ∈ {(6,12,24), (8,16,32), offset_10}
n_jitter      ∈ {3,4}
cross_family  ∈ {core, core_plus_segment}
```

禁止：

- 随机生成 50 个世界后在同一 OOF 上挑最高者；
- 每个候选同时改归一化、模型深度、树数和 jitter；
- 全配对扩到 80～170 个类别交叉；
- 因为某个候选差 0.0003 就微调分箱数追分；
- 把弱但解耦的世界直接放进 `max`。

## 5. 两级筛选

### 5.1 Discovery：4 seeds

每个候选和 main/alt 父臂使用同一组：

```text
seeds = 20400, 20401, 20402, 20403
folds = S2 的胜出折数；若 S2 未完成，固定 5
```

每个候选报告：

```text
arm_auc
delta_vs_paired_main
corr_vs_main
corr_vs_alt
max_fusion_delta
mean_fusion_delta
positive_seed_count
```

晋级规则：

- 强度型：相对 paired main `ΔAUC ≥ 0`，且融合增益 ≥ +0.001；
- 多样性型：单臂不低于 main `0.0015` 以上、最大相关 ≤0.95、融合增益 ≥ +0.001；
- 两类都要求至少 3/4 seeds 方向不矛盾。

最多保留 4 个，不能按裸 AUC 排名前 4；应保留强度—相关性的 Pareto 前沿。

### 5.2 Confirmation：4 个未见 seeds

冻结候选定义后运行：

```text
seeds = 20410, 20411, 20412, 20413
```

最终保留条件：

- confirmation 的融合增益为正；
- 8 seeds 合并后配对 `P(Δ>0) ≥ 90%`；
- 单臂相对 main 不低于 `-0.0015`；
- 与已选世界的最大相关 ≤0.96；
- 最多留下 2～3 个新世界。

只在 discovery 上高、confirmation 回落的世界直接淘汰，不进行“再换一组 seed”挽救。

## 6. 融合规则

新世界先逐个加入，规则预登记：

```text
base_max       = max(main_d5, main_d6, alt)
base_mean      = 0.25*main_d5 + 0.25*main_d6 + 0.50*alt
plus_new_max   = max(base members, new)
plus_new_mean  = 0.80*base_mean + 0.20*new
two_new_mean   = 0.70*base_mean + 0.15*new1 + 0.15*new2
```

使用限制：

- 新臂与最强臂差距 ≤0.0015 才可进入 `max`；
- 差距在 0.0015～0.003 时只允许 10%～20% rank mean；
- 差距 >0.003 时无论相关性多低都淘汰；
- 最终仍由 5 块嵌套评估，不看全量 OOF 临时改权重。

## 7. 实现结构

建议将世界定义参数化，而不是继续复制 `build_alt3/alt4`：

```text
WorldSpec
  name
  normalization
  bin_scheme
  cross_family
  n_jitter
  jitter_bins
  stream_offset
```

统一入口：

```bash
PYTHONPATH=src2 python3 src2/run_oof.py \
  --world-spec configs/worlds/robust_source.json \
  --arms cat_world \
  --seeds 20400 20401 20402 20403 \
  --out artifacts/gpt56/world_robust_source_discovery
```

每个配置文件在运行前提交到 Git；配置 hash 写入产物 manifest，防止结果出来后无痕改参数。

## 8. 预期收益与成本

| 结果层级 | 合理预期 |
|---|---:|
| 修复 alt2 单臂 | +0.001～+0.003（相对原 alt2） |
| 一个合格新世界加入融合 | +0.0008～+0.002 |
| 两到三个确认通过的新世界 | 累计 +0.0015～+0.004 |

增益不会线性相加。第二、第三个世界的边际收益会递减，且世界越多，融合选择方差越大。

单候选 4 seeds 的历史成本约 12～20 分钟；8 个 discovery 候选约 2～3 小时，
confirmation 与最终补种子约 1.5～2.5 小时。可以并行跑候选，但必须使用独立输出目录。

## 9. 失败时怎么走

- 全部世界单臂弱：说明 v2 编码已接近该范式上限，转 S5 正交臂；
- 单臂变强但相关 >0.98：替换父臂，不叠加；
- 单臂略弱但相关 <0.94、mean 融合升：作为 10%～20% 多样性臂；
- discovery 升、confirmation 不升：判定选择噪声，不扩大搜索；
- alt2-repair 仍弱：停止 region×source 标准化，不继续调收缩常数。

## 10. 完成定义

- alt2 的弱点有单变量修复对照；
- discovery 不超过 12 个预登记候选；
- confirmation 使用未参与筛选的 seeds；
- 最终只保留 Pareto 前沿上的 2～3 个世界；
- 每个保留世界有配置、manifest、逐 seed 预测和融合差值报告；
- 不覆盖 v2，单独产出 S3 候选。

