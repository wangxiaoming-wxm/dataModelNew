# S1：v2 与冻结 B7 的探索性融合

## 1. 假设

`submission_v2` 和 B7 都以 CatBoost 为主，但特征视图、训练协议和融合成员不同。
两者公开榜分别为 `0.70878` 和 `0.70722`，都具备可用强度。如果排序错误不完全重合，
在不训练新模型的情况下融合两套冻结预测可能提高 AUC。

本步骤不替换 v2。v2 仍是主基线。B7 只有重建出诚实 OOF 后才能成为正式多样性臂；
直接使用现有冻结 B7 只能作为探索性提交。

## 2. 已有证据

### 2.1 测试预测的互补性

| 指标 | 数值 |
|---|---:|
| 两个提交的 Spearman 相关 | 0.96763 |
| Pearson 相关 | 0.84646 |
| 平均绝对名次差 | 344.4 / 6398 |
| v2 公开榜 | 0.70878 |
| B7 公开榜 | 0.70722 |

相关性足够高，说明 B7 不会完全改变 v2 的排序；又没有高到 0.99，仍有可能提供互补信息。

### 2.2 冻结 OOF 回测

为保证 OOF 与 test 使用完全相同的变换，先对 v2 的 `views_max` 输出整体重排，再对
`artifacts/b7_closest/predictions.npz` 的冻结 OOF 重排：

| 规则 | OOF AUC | 相对 v2 |
|---|---:|---:|
| v2 | 0.69910 | — |
| `0.75*v2 + 0.25*B7` | 0.70145 | +0.00235 |
| `0.50*v2 + 0.50*B7` | 0.70285 | +0.00375 |
| `max(v2, B7)` | 0.70358 | +0.00448 |

这组数字只说明“两个冻结系统的预测可能互补”，不能作为正式晋级证据。

### 2.3 必须折价解释

上表不能直接解释为榜单必涨 0.004：

- B7 OOF 使用了外层验证折早停；`+0.00254` 只是在一个 gap 单种子对照中测得，
  不能当作整个 B7 多臂系统的确定偏差；
- B7 的最终 OOF 还是多臂融合结果，不能把 bootstrap 区间当成重新训练后的不确定性；
- v2 与 B7 的两次榜单分已经被看到，继续反复试权重会开始拟合公开榜；
- 两个系统的 test Spearman 仍高达 0.968，真实增益可能明显小于 OOF 表面值。

对有偏冻结 OOF 做 bootstrap 也无法消除早停和候选选择偏差。因此该实验的保守预期只能写成
“方向未知、可能有小幅增益”，不能给出可靠的榜单增益区间。

### 2.4 正式使用前的诚实重建

冻结保留 B7 最终配方中的 `gap`、`gap_bag`、`plus` 三个成员，不在重建后增删臂。
执行前实现：

```bash
python3 scripts/gpt56_rebuild_b7_honest.py \
  --arms gap gap_bag plus \
  --outer-folds 5 --inner-folds 5 \
  --seeds 20260 20261 20262 20263 \
  --out artifacts/gpt56/s1_b7_honest
```

对每个 outer fold，只在 outer-training 的 inner folds 上早停，取各 inner fold
`best_iteration` 中位数，然后用该固定树数在完整 outer-training 重训并预测 outer-valid。
test 预测使用同一批已冻结树数规则。脚本必须产出
`artifacts/gpt56/s1_b7_honest/predictions.npz` 和逐折树数报告。

正式候选必须满足：

1. 与 v2 使用相同 outer folds；
2. 按上述 inner-fold 中位数规则选树数；
3. outer-valid 只生成一次预测，不参与早停、选臂或选权重；
4. 重新生成 test 预测和 B7 融合；
5. 用该诚实 OOF 重算本文件全部表格。

无法完成重训时，S1 不进入最终本地候选池。

## 3. 固定候选，不做连续权重搜索

在新提交前冻结：

```text
primary   = 0.50 * rank(v2) + 0.50 * rank(B7)
fallback  = 0.75 * rank(v2) + 0.25 * rank(B7)
explore   = max(rank(v2), rank(B7))
```

规则说明：

- 主候选用 50/50：OOF 增益接近 max，但算子更平滑；
- 保守候选用 75/25：更多保留已获得 `0.70878` 的 v2；
- 在一致重排后 `max` 与 50/50 差约 `0.00073`，仍须由诚实 B7 OOF 决定，不能依据有偏表优先；
- 不再测试 0.1 间隔的权重网格，也不根据榜单结果改成 0.47/0.53。

## 4. 生成步骤

### 4.1 需先实现的入口

当前仓库没有冻结融合脚本。执行前新增并测试：

```text
scripts/gpt56_frozen_blend.py
  --v2 submissions/submission_v2.csv
  --b7 submissions/submission_b7_closest_honest.csv
  --v2-oof-dir artifacts/v2
  --b7-oof artifacts/gpt56/s1_b7_honest/predictions.npz
  --sample data/submit_sample.csv
  --out-dir submissions/
  --report artifacts/gpt56/s1_frozen_blend/report.json
```

该接口是实现规格，不是当前已经可运行的命令。

### 4.2 输入

```text
submissions/submission_v2.csv
submissions/submission_b7_closest_honest.csv
artifacts/v2/arm_cat_*.npz
artifacts/b7_closest/predictions.npz
data/submit_sample.csv
```

### 4.3 强制按 ID 对齐

不能假设两个 CSV 当前行序永远一致。生成脚本必须：

1. 分别检查 ID 唯一；
2. 以 `submit_sample.csv` 的 ID 顺序左连接；
3. 检查无缺失、无多余 ID；
4. 对每个预测列独立做 `(rank - 0.5) / n`；
5. 再按冻结规则融合。

两个提交的数值尺度差异很大：

- v2 是接近 `[0.001, 0.999]` 的秩分数，均值约 0.531；
- B7 是概率风格分数，范围约 `[0.023, 0.883]`，均值约 0.107。

直接平均原值会让 v2 尺度主导，必须先转秩。ROC-AUC 只关心顺序，最终无需概率校准。

### 4.4 输出

```text
submissions/gpt56_s1_frozen_blend.csv
submissions/gpt56_s1_v2_75_b7_25.csv
submissions/gpt56_s1_rankmax_v2_b7.csv
artifacts/gpt56/s1_frozen_blend/report.json
```

`report.json` 保存输入 SHA256、相关性、OOF 表、bootstrap 区间、规则和输出 SHA256。

## 5. 晋级与提交决策

进入最终候选池需满足：

- 三套输入的 y 完全一致；
- B7 每个成员都具有无外层早停/选择的诚实 OOF；
- 诚实 OOF 的 50/50 融合增益 ≥+0.0015；
- 描述性 bootstrap 90% 区间下界 >0；
- test 排序与任一父提交的 Spearman >0.95，避免生成错误；
- 提交格式通过
  `src2/verify.py --submission submissions/gpt56_s1_frozen_blend.csv --out artifacts/gpt56/s1_frozen_blend/verify.json`；
- 生成逻辑不使用任何新的榜单反馈。

只有上述条件完成时，50/50 才能进入正式候选。若未诚实重训但仍有独立的探索提交预算，
可以提交 `gpt56_s1_frozen_blend.csv`，但不得用结果修改权重或决定后续是否保留 B7。

如果只能使用一个提交名额，跳过探索性 S1，把名额留给 S2/S3 后的本地冠军。

## 6. 如何解释榜单结果

- 只记录分数、文件 hash 和平台解析是否正常；
- 上涨、持平或下跌都不改变预先冻结的本地候选集合；
- 是否保留 B7 只由诚实重训后的本地协议决定；
- 本步骤最多消耗 1 个探索提交，不用第二次提交反推精细权重。

## 7. 完成定义

- 冻结融合脚本可重复生成三个候选；
- 有输入/输出哈希和“有偏冻结 OOF”醒目标记；
- 正式使用前完成 B7 诚实重训；否则只允许探索提交；
- 榜单结果不影响是否保留历史臂，也不反向调权重。

