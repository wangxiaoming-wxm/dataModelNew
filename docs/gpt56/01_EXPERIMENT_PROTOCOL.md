# S0：实验协议与可信基线

## 1. 目的

当前最大风险不是模型不够复杂，而是 14930 行数据上的 AUC 波动足以掩盖 `0.001～0.003`
的真实增益。S0 要先让每个候选都能回答：

> 在相同样本、相同折、相同随机种子下，这个改动比 v2 好多少，差值有多稳定？

只有完成本步骤，后面的 10 折、编码世界和融合结果才有决策价值。

## 2. 冻结基线

### 2.1 不重训正式基线模型的复核

以下命令均从仓库根目录执行。当前 `requirements.txt` 未声明 `lightgbm`，但 `arms.py`
会无条件导入它；在修复依赖声明或改成懒加载前，需要显式安装最新版：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install lightgbm
mkdir -p artifacts/gpt56/s0
cp -a artifacts/v2 artifacts/gpt56/s0/v2_repro
PYTHONPATH=src2 python3 src2/fuse.py \
  --dir artifacts/gpt56/s0/v2_repro \
  --submission submissions/submission_v2_reproduced.csv
PYTHONPATH=src2 python3 src2/verify.py \
  --submission submissions/submission_v2_reproduced.csv \
  --out artifacts/gpt56/s0/verify.json
sha256sum submissions/submission_v2.csv submissions/submission_v2_reproduced.csv
```

复制 artifact 是为了避免 `fuse.py` 覆盖冻结的 `artifacts/v2/fusion_report.json`。
`verify.py` 会为打乱标签训练 5 个 sanity-check 模型，但不会重训正式 v2 基线。

必须复核：

- 嵌套 OOF：`0.6985627496`；
- 提交规则：`views_max`；
- 行数：6398；
- ID 顺序与 `data/submit_sample.csv` 一致；
- 当前 main/cat_d5 打乱标签 sanity check AUC：0.47～0.53；
- 原提交与复算提交内容一致。

### 2.2 冻结元数据

新增每次运行的 `manifest.json`，至少保存：

```json
{
  "experiment_id": "s2_10fold_seed20260",
  "parent": "v2",
  "git_commit": "<sha>",
  "train_sha256": "<full_64_char_sha256>",
  "test_sha256": "<full_64_char_sha256>",
  "view": "main",
  "arms": ["cat_d5"],
  "folds": 10,
  "seeds": [20260],
  "stream_base": 0,
  "fixed_iterations": true
}
```

缺少 manifest 的产物不得进入最终融合。

## 3. 补齐当前流水线丢失的信息

`src2/run_oof.py` 当前只保存所有种子聚合后的 `oof/test`，在第 62～70 行已经丢掉：

- 每个种子的原始概率；
- 每个折的 test 原始概率；
- 每行属于哪个验证折；
- 秩平均前的 OOF 概率。

这会导致“概率平均 vs 秩平均”“树数随机化是否增加多样性”等问题无法回测。后续实现时，
每个臂额外保存：

```text
oof_prob_by_seed       [n_seed, n_train]
oof_rank_by_seed       [n_seed, n_train]
test_prob_by_model     [n_seed, n_fold, n_test]
test_rank_by_model     [n_seed, n_fold, n_test]
fold_id_by_seed        [n_seed, n_train]
y                      [n_train]
```

聚合后的旧字段继续保留，保证现有 `merge_runs.py`、`collect.py` 和 `fuse.py` 可用。
在实现前，必须同步修改 `merge_runs.py`：逐 seed 数组沿 seed 轴拼接，逐模型 test 数组保留
`seed×fold` 两层元数据；只合并完整 seed。OOF 按完整 seed 数加权，test 按实际 fold 模型数加权，
二者不能共用一个“模型数权重”。

除明确标记为 `test_spearman` 的生成健全性检查外，模型多样性阈值统一指
**OOF 经验秩的 Spearman 相关**。

### 3.1 transductive 变换边界

在 train+test 上拟合无标签分箱、组中位数和秩是 transductive preprocessing，不是目标泄漏，
但必须先确认比赛规则允许。每类新变换都补一个 `outer-train-only` 拟合对照：

- 若与 transductive 版本差异在 ±0.001 内，优先使用更易解释的 outer-train-only；
- 若 transductive 明显更好，报告差异并在规则允许的前提下使用；
- 无论哪种版本，监督统计量都只能在对应训练折内拟合。

## 4. 配对比较方法

### 4.1 强制共用切分

普通特征/模型实验的父基线和候选必须共享：

- `StratifiedKFold` 的 `n_splits`；
- 完全相同的 seed；
- 完全相同的验证行索引；
- 相同的 jitter stream，除非 jitter 本身就是唯一自变量；
- 相同模型预算，除非训练量就是实验变量。

禁止拿 `artifacts/A` 的裸 AUC 与另一批随机种子的 `artifacts/B` 裸 AUC直接比较。
S2 是唯一变量为折数的特例：5/10 折不可能共享 `n_splits` 或验证索引，只要求相同数据行、
seed、jitter、模型参数和预先登记的 splitter seed，并对两份完整 OOF 做行级配对。

### 4.2 四层指标

按下面顺序判断：

1. **主指标**：所有重复聚合后的 OOF AUC；
2. **配对差值**：`AUC(candidate) - AUC(parent)`；
3. **重复一致性**：每个 seed 的差值为正的比例；
4. **配对分层 bootstrap**：正负样本分别有放回抽样，计算 2000 次 AUC 差值。

报告：

```text
delta_point
delta_ci90
delta_ci95
bootstrap_positive_fraction
positive_seed_count / total_seed_count
spearman_vs_parent
```

bootstrap 必须对两个预测使用同一组抽样索引；分别 bootstrap 再相减会夸大方差。
`bootstrap_positive_fraction` 只是“给定当前冻结 OOF 预测后的重采样正差比例”，不是模型真实改善概率；
它不覆盖训练集扰动、CV 依赖或候选选择。

### 4.3 已完成的 v2 融合风险复核

在现有 OOF 上按 3199 行模拟半榜、2000 次 bootstrap：

| 规则 | OOF AUC | bootstrap 标准差 | 5% 分位 |
|---|---:|---:|---:|
| `views_half` | 0.69716 | 0.01519 | 0.67174 |
| `views_max` | 0.69910 | 0.01517 | 0.67362 |

`views_max - views_half` 的均值为 `+0.00195`，5%～95% 区间约
`[-0.00079, +0.00476]`，bootstrap 正差比例为 86.4%。因此旧文档担心 `max` 的抽样标准差
会高 30% 并未发生；真正的不确定性是两条规则差异本身较小，而不是 `max` 明显更不稳定。

## 5. 两级实验制度

### 5.1 发现集

- 5 折 × 4 个固定 discovery seeds；
- 只用于淘汰明显无效候选；
- 编码世界最多同时筛 12 个；
- 晋级：`ΔAUC ≥ +0.0015`；
- 多样性臂可允许单臂略弱，但其冻结融合必须 `ΔAUC ≥ +0.0008`，
  且与父模型 OOF 经验秩 Spearman ≤0.94。

### 5.2 确认集

- 使用未参与筛选的 4 个 confirmation seeds；
- 参数、特征和融合规则全部冻结后运行；
- 通过标准：
  - 确认集方向为正；
  - discovery + confirmation 合并后 bootstrap 正差比例 ≥90%；
  - 至少 6/8 个 seed 的差值为正；
  - 没有数据校验、格式或候选级打乱标签 sanity-check 异常。

seed 隔离只能检查随机初始化稳定性，因为两组 seed 仍复用同一批标签。大量候选中选最好者后直接报
同一批 OOF 仍会产生选择乐观。正式无偏评估需要最外层按行划分，并在每个 outer-train 内重新完成
候选筛选、参数选择和融合规则选择；outer-valid 只允许评分。若不支付这部分重训成本，最终结果必须
标为“给定已选候选后的条件评估”，不能称为消除了全部选择乐观。

### 5.3 全局随机数登记

| 用途 | seeds |
|---|---|
| S2 discovery | 20260, 20261；扩展确认 20262, 20263 |
| S2 confirmation | 20264, 20265, 20266, 20267 |
| S3 discovery | 20400～20403 |
| S3 confirmation | 20410～20413 |
| S4 discovery | 20500～20503 |
| S4 confirmation | 20510～20513 |
| S4 组合确认 | 20520～20523 |
| S5 discovery | 20600～20603 |
| S5 confirmation | 20610～20613 |
| 最终 meta-fold | `random_state=20999` |

同一阶段若需要配对对照必须使用同一行 seeds；不同阶段不得借“换 seed”挽救失败结果。

## 6. 融合评估

### 6.1 规则预登记

每轮融合最多登记 6 条规则，建议固定为：

1. v2 原始 `views_max`；
2. 所有强世界的等权 rank mean；
3. 所有强世界的 elementwise rank max；
4. `0.75 * v2 + 0.25 * new`；
5. `0.50 * v2 + 0.50 * new`；
6. `0.90 * v2 + 0.10 * orthogonal_arm`。

不允许看到全量 OOF 后追加 `0.63/0.37` 之类的精细权重。
S6 的 R0～R5 是唯一权威最终规则表；本节只用于阶段内筛选。

### 6.2 嵌套决策

对已冻结候选做条件融合评估时，外层 5 块，在 4 块上选规则，在第 5 块上评分。除了
`conditional_nested_oof_auc`，还要记录：

- 每条规则被选中的次数；
- held-out 块上的逐块增量；
- 最优规则与第二名的差值；
- 规则选择是否稳定。

若 5 块不能至少 4 块选中同一规则，使用预登记的第 5 条 50/50 简单规则，
而不是结果出来后新增回退规则。此层只控制融合规则选择，不控制上游候选开发的乐观。

## 7. 实验台账与停止规则

建议用 `artifacts/gpt56/registry.csv` 统一登记：

```text
id,parent,hypothesis,commit,status,folds,seeds,oof,delta,ci90_low,ci90_high,positive_seeds,corr,cost_sec,decision,notes
```

停止规则：

- 同一方向连续 3 个候选未达到快筛门槛，停止该方向；
- 只改变常规 CatBoost 超参且增益 <0.001，不补种子；
- 弱臂单臂 AUC <0.685，不进入融合搜索；
- 新编码世界比主臂弱 >0.003，即使相关性低也不能进 `max`；
- 不因一次公开榜下降修改参数；只有新的本地证据才能开启下一轮。

## 8. S0 完成定义

执行最终候选前需实现候选级审计接口：

```bash
python3 scripts/gpt56_shuffle_audit.py \
  --config configs/gpt56/final_rules.json \
  --shuffle-seed 20888 \
  --out artifacts/gpt56/final/shuffle_audit.json
```

脚本必须从打乱后的 y 开始，重建候选使用的全部监督变换、基础臂和融合；报告配置 hash、代码 commit、
各臂和最终融合 AUC。当前该脚本不存在，是 S0 的实现前置。

- v2 可从冻结产物完全复算；
- 新产物保存逐种子、逐折原始概率和切分；
- 有统一的配对 bootstrap 描述性报告；
- 有 discovery / confirmation seeds 的固定清单；
- 有机器可读 registry；
- 后续每个实验都能追溯到 commit、数据哈希和父基线；
- 候选级 shuffle harness 会从打乱后的标签开始重建全部监督变换和融合；当前 `verify.py`
  只作为 main/cat_d5 sanity check，不能证明新世界、TE/GLM 或最终融合无泄漏；
- 每个候选另做静态数据流审计，确认标签只在对应训练折内进入监督变换。

