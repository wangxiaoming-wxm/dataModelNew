# 特征剪枝（Feature Pruning）探测结论

## 动机

对方在证伪 Lossguide 后提出：去掉弱特征（与过往「加特征」方向相反）可能有增益。

## 协议

- 世界：`build_main`（116 特征）
- 模型：CatBoost Plain d5 l2=10（烟测；非全量 Ordered 8-seed）
- 重要性：`PredictionValuesChange`
- 保留比例：100%→50%；另测家族删除（freq / bottom / core）
- 对照锚点：fp_v5 OOF **0.72880**

## 结果

### 按重要性保留（相对全特征烟测）

| keep | k | OOF Δ | nested fold-mean Δ | 与 fp_v5 融合 Δ |
|---:|---:|---:|---:|---:|
| 0.9 | 104 | −0.0009 | −0.0015 | −0.0065 |
| 0.7 | 81 | ≈0 | +0.0004 | −0.0065 |
| **0.5** | **58** | **+0.0019** | **+0.0017** | **−0.0059** |

### 家族删除

`drop_freq` / `drop_bottom20` / `core_keep`：solo **无稳定正增益**；与 fp_v5 融合一律为负。

### 与冻结 main 相关性

剪枝臂 Spearman(main) ≈ **0.95** → 基本是同构信号，已被 AM40/fp_v5 吸收。

## 判决

**SOLO_LIFT_ONLY → 不晋级冠军。**

- keep=50% 对「单臂 CatBoost」有真实小增益（nested ≈ +0.0017）
- **无法抬升 fp_v5**：融合 Δ 约 −0.006
- 若未来重训主臂，可把 keep≈50–60% 当作可选烟测；门槛须融合 OOF **> fp_v5**

## 弱特征观察（重要性底）

`t1/r1/condition_missing`≈0；`w1/w2/c1/c2`、多数 `f_*` 频次极弱。强特征集中在 `rl/sc10a/ratio/rsa/c10s/...`（days–condition–region/source 交叉）。
