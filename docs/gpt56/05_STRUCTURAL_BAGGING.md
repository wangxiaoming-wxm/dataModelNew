# S4：结构化 Bagging

## 1. 目标

普通 seed bagging 已在 8 个以后饱和。S4 不再增加同构模型数量，而是检验三种结构多样性：

1. 保留原始概率中的置信度信息；
2. 预先分配不同树数，模拟早停带来的容量多样性但不偷看验证折；
3. 让不同模型各看一部分 jitter 编码，而不是在同一模型中堆更多列。

三项必须分开做消融，不能一次全部打开。

## 2. E1：概率平均与秩平均

### 2.1 当前行为

`src2/run_oof.py` 当前：

- 每个 seed 的完整 OOF 先转秩；
- 每个 fold 的 test 预测也先转秩；
- 然后做平均。

这样能消除不同模型的校准差异，但也可能丢掉“某行风险远高于相邻行”的置信度。
现有 `max` 融合发生在经验秩上，不能作为原始概率置信度有价值的证据；E1 纯属待验证假设。

### 2.2 预登记聚合器

完成 S0 保存后，只比较：

```text
A current_rank
  OOF  = mean(rank(oof_probability_by_seed))
  test = mean(rank(test_probability_by_fold_model))

B raw_probability
  OOF  = rank(mean(oof_probability_by_seed))
  test = rank(mean(test_probability_by_fold_model))

C logit_standardized
  每个模型先 clip 到 [1e-5, 1-1e-5]，转 logit，
  再用该模型预测分布的 median/MAD 做无标签标准化，最后平均并转秩
```

不做 Platt/isotonic 校准：AUC 只关心排序，且在同一 OOF 上拟合监督校准器很容易产生乐观。

### 2.3 设计和门槛

- 现有 artifacts 没有原始概率，需先按 S0 新 schema 重跑一次基础模型；
  得到这批原始预测后，不为 A/B/C 三个聚合器重复训练；
- `cat_d5`、`cat_alt` 分别比较，不能只看最终融合；
- 报告每 seed 差值、聚合 OOF、与 current 的相关性；
- 聚合器在 discovery seeds 上选定后，用 confirmation seeds 验证；
- 晋级：确认集 `ΔAUC ≥ +0.001`，或最终融合 `≥ +0.001` 且单臂不降超过 0.001。

若 B/C 只在一个臂上有效，则只改变该臂。没有理由要求所有臂使用相同聚合器。

## 3. E2：预登记树数日程

### 3.1 假设

旧 B7 的外层早停存在统计乐观，但它还让每折模型拥有不同树数，可能给 bagging 增加结构多样性。
可以把“树数不同”与“偷看验证标签”分离：树数在训练前由固定日程决定。

### 3.2 固定计算量的日程

`cat_d5` 当前每模型 1000 树。比较：

```text
fixed     = [1000, 1000, 1000, 1000]
scheduled = [700, 900, 1100, 1300]
```

两组平均都是 1000。运行前按全部 `(seed, fold)` 生成平衡 Latin schedule，
确保四种树数出现次数之差不超过 1，并在 manifest 记录实际总树数。这样才能确保：

- 不读取验证 AUC；
- 重跑完全确定；
- 实际总树数和计算预算一致（允许不足一个模型树数的舍入误差）；
- 容量范围足以制造差异，但不包含极端弱模型。

`cat_d6` 只有在 d5 通过后再试：

```text
fixed     = [700, 700, 700, 700]
scheduled = [500, 650, 750, 900]  # 平均 700
```

### 3.3 实验矩阵

```text
view  = main
arm   = cat_d5
folds = S2 胜出折数
discovery seeds    = 20500..20503
confirmation seeds = 20510..20513
```

只修改 `iterations`。先比较：

- fixed bag vs scheduled bag；
- 成员平均相关性；
- 按树数分组的“同 seed/同 fold scheduled AUC − fixed AUC”局部诊断；
- 与 alt 臂融合后的增益。

晋级：

- scheduled bag 确认集增益 ≥ +0.001；
- 至少 6/8 seeds 方向为正；
- 任一树数组的平均配对 fold 差值不能低于 -0.004；
- 提升不能只来自一次幸运的 1300 树模型。

不通过就停止，不根据结果把范围改成 742～1261。

## 4. E3：模型间 jitter 分工

### 4.1 现有证据

- 同一模型使用 4 套 jitter 优于 2 套；
- 堆到 6 套反而约下降 0.002；
- 每个 seed 使用不同 jitter 流有收益。

这表明“编码平均”有效，但单模型列数过多会增加过拟合。新假设是：让两个模型各看较少且不重叠的
jitter，再平均模型输出，可能比一个模型同时看 4～6 套列更稳。

### 4.2 先做配对机制试验，再做固定预算比较

Discovery 先用完全相同的 4 个 base seeds 比较：

```text
A_pair：4 seeds × 1 model × 4 jitter views = 4 models/fold
B_pair：4 seeds × 2 submodels × 2 disjoint jitter views = 8 models/fold
```

B 中每个 seed：

```text
submodel_1 streams = [base+0, base+1]
submodel_2 streams = [base+2, base+3]
seed prediction    = mean(rank(submodel_1), rank(submodel_2))
```

这组可以逐 seed 配对，但 B 使用两倍模型数，所以只回答“额外算力用于 jitter 分工是否有效”。
通过后再做固定预算的非配对生产对照：

```text
A_budget：8 seeds × 1 model × 4 jitter views = 8 models/fold
B_budget：4 seeds × 2 submodels × 2 jitter views = 8 models/fold
```

8 个 A seeds 固定为 `20500..20503 + 20510..20513`，B 使用其中 discovery 的前 4 个。
该固定预算对照 seed 数不同，不能称为配对。不测试更多组合。

### 4.3 门槛

- `B_pair` 相对 `A_pair` 的逐 seed 配对增益 ≥ +0.001；
- `B_budget` 相对 `A_budget` 的非配对结果不能下降；
- 子模型相关性应明显低于普通相邻 seeds；
- confirmation seeds 方向为正；
- 加入整体融合后不被 main/alt 相关性抵消；
- 计算量和峰值内存有明确记录。

## 5. 实现前置

当前仓库尚不保存逐模型原始概率，也没有树数 schedule 或 submodel-jitter CLI。执行前需实现：

```text
src2/run_oof.py
  --save-raw-predictions
  --iteration-schedule <json>
  --jitter-submodels <int>
  --jitter-views-per-submodel <int>

scripts/gpt56_compare_aggregators.py
scripts/gpt56_structural_report.py
```

脚本统一读取 S0 schema，输出到 `artifacts/gpt56/s4_*`；这些名称是接口规格，不代表当前已经存在。

## 6. 实验顺序

严格按成本从低到高：

1. E1：只重算已有预测，秒级；
2. E2：先 4 seeds，约一次主臂训练成本；
3. E3：固定 8 模型/fold 的 A/B；
4. 每项单独确认；
5. 最后只组合已经独立通过的项，并使用预登记的 `[20520,20521,20522,20523]` 做组合确认。

组合确认必须再做一次 4 个新 seeds。单项各自 +0.001 不代表组合一定 +0.002，
因为它们都可能通过增加同一种方差多样性起作用。

## 7. 与 S2/S3 的关系

- S4 discovery 优先在 5 折 main 上做，成本最低；
- 如果 S2 证明 10 折有效，只把 S4 胜出的机制迁移到 10 折做一次确认；
- 如果 S3 已得到新强世界，先保持其聚合方式不变，S4 机制只在 main 上确认后再推广；
- 不允许“新世界 + 10 折 + 随机树数 + 概率平均”一次上线，否则无法判断收益来源。

## 8. 不做的后处理

- 不调分类阈值：ROC-AUC 与阈值无关；
- 不做概率校准作为提分手段：任何严格单调校准都不会改变 AUC；
- 不做伪标签：0.70 AUC 下会放大自身错误；
- 不按公开榜结果挑 rank/prob 聚合器；
- 不在验证折早停，即使只把 best iteration 用作“多样性”也不允许。

## 9. 预期收益

| 子实验 | 合理预期 | 成本 |
|---|---:|---:|
| E1 聚合器 | 0～+0.0015 | 秒级回测 |
| E2 树数日程 | 0～+0.002 | 约 30～60 分钟 |
| E3 jitter 分工 | +0.0005～+0.002 | 约 1～2 小时 |

总增益不能线性相加。S4 的成功标准是找到一个可复现的 `+0.001～+0.002` 机制，
而不是把三个小波动全部塞进最终模型。

## 10. 完成定义

- 三项消融分别有 paired report；
- 聚合器选择使用冻结的 discovery/confirmation；
- 树数日程与 fixed 计算预算一致；
- jitter 分工与普通 seeds 在相同模型数量下比较；
- 最终最多保留 1～2 个结构 bagging 改动；
- 所有最终模型仍满足固定参数、严格折外预测。

