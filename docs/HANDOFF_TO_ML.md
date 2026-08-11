# SUPER714 给 ML 工程师的交接

## 1. 任务边界

需要由 ML 工程师完成的唯一新训练是第三臂 `main_te`。不要重写 best v1 两个冠军臂，不要增加训练框架、调参搜索、其他 TE 或旧臂。

输入基座：

- 代码：`real714_pkg/v1_best714/src/explore_best.py`
- 冠军 OOF：`real714_pkg/v1_best714/artifacts/best_oof.npy`
- 冠军 test：`real714_pkg/v1_best714/artifacts/best_test.npy`
- 新 FE：`src_super/features_te.py`

## 2. 三臂精确设定

### A. `best_main`（直接复用）

```text
features       = explore_best.build_main
model          = CatBoostRegressor
loss/eval      = RMSE / RMSE
boosting_type  = Ordered
depth          = 5
iterations     = 800
learning_rate  = 0.03
l2_leaf_reg    = 10
random_strength= 0.7
outer folds    = StratifiedKFold(5, shuffle=True, random_state=seed)
seeds          = 2026,2027,2028,2029,2030,2031,2032,2033
bag seeds      = 0,1,2
model seed     = seed*100 + bag_seed
early stopping = 禁止
```

验收锚点：单臂 pooled OOF **0.699917**。

### B. `best_alt`（直接复用）

除以下差异外与 A 相同：

```text
features       = explore_best.build_alt
boosting_type  = Plain
depth          = 6
iterations     = 800
l2_leaf_reg    = 6
```

验收锚点：单臂 pooled OOF **0.697704**。

### C. `main_te`（唯一新臂）

基础特征帧、CatBoost 参数、outer fold、seed 和 bagging 全部复制 A，只新增一个**数值列**：

```text
name           = te_source_days_bin10
key            = source × days_bin10
days bins      = 每个外层训练折单独拟合的 10 等频桶
smoothing      = 20.0
inner folds    = StratifiedKFold(4, shuffle=True, random_state=outer_seed)
unseen key     = 回退到相应映射训练集的 y.mean()
```

不要把 TE 列加入 `cat_features`。禁止 `use_best_model`、`eval_set` 早停、额外 TE 列、更多 bins/smoothing 候选。

## 3. 每个外层 fold 的调用约定

best 的 label-free 主帧仍按原代码生成；TE 本身必须按外层 fold 重新生成：

```python
from src_super.features_te import FEATURE_NAME, build_source_days_te

te_fold = build_source_days_te(
    fit_frame=raw_train.iloc[tri],
    y_fit=y[tri],
    valid_frame=raw_train.iloc[vali],
    other_frames=(raw_test,),
    n_bins=10,
    smoothing=20.0,
    inner_splits=4,
    inner_seed=seed,
)

Xtr = X_main_train.iloc[tri].copy()
Xva = X_main_train.iloc[vali].copy()
Xte = X_main_test.copy()
Xtr[FEATURE_NAME] = te_fold.fit.to_numpy()
Xva[FEATURE_NAME] = te_fold.valid.to_numpy()
Xte[FEATURE_NAME] = te_fold.others[0].to_numpy()
```

关键不变量：

1. `te_fold.fit` 是内层 OOF 值，不能用完整外层训练统计覆盖；
2. `te_fold.valid` 与 test 只来自外层训练折完整统计；
3. days 边界只能从 `raw_train.iloc[tri]` 拟合；
4. validation/test 的标签永远不传入 FE；
5. 同一个 outer fold 的三个 bag 共用同一份 TE，避免无意义的 FE 搜索。

## 4. rank-pool 与 test 聚合

严格复制 best v1：

```python
from scipy.stats import rankdata

# 每个 seed：5 折 OOF 已回填；test 已先平均 5 fold × 3 bag
oof_seed_rank = rankdata(oof_seed) / len(oof_seed)
test_seed_rank = rankdata(test_seed) / len(test_seed)

# 八个 seed 等权
main_te_oof = np.mean(oof_seed_ranks, axis=0)
main_te_test = np.mean(test_seed_ranks, axis=0)
```

建议保存：

```python
np.savez(
    "artifacts/super714_main_te.npz",
    oof=main_te_oof,
    test_pred=main_te_test,
    per_seed=np.asarray(seed_aucs),
    seeds=np.arange(2026, 2034),
    y=y,
)
```

## 5. 融合代码约定

冠军文件中的 `main/alt` 已经是同口径 rank-pool，不要重新训练权重：

```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

best_oof = np.load(
    "real714_pkg/v1_best714/artifacts/best_oof.npy",
    allow_pickle=True,
).item()
best_test = np.load(
    "real714_pkg/v1_best714/artifacts/best_test.npy",
    allow_pickle=True,
).item()

main_oof, alt_oof = best_oof["main"], best_oof["alt"]
main_test, alt_test = best_test["main"], best_test["alt"]

max2_oof = np.maximum(main_oof, alt_oof)
max3_oof = np.maximum.reduce([main_oof, alt_oof, main_te_oof])
max3_test = np.maximum.reduce([main_test, alt_test, main_te_test])

te_auc = roc_auc_score(y, main_te_oof)
corr_main = spearmanr(main_te_oof, main_oof).statistic
base_auc = roc_auc_score(y, max2_oof)
super_auc = roc_auc_score(y, max3_oof)
gain = super_auc - base_auc
```

只允许无权重 `max3`。不要搜索 mean、power、stack、臂子集或权重。

## 6. 强制验收数字

先复核冻结基座：

| 检查 | 预期 |
|---|---:|
| `AUC(main)` | 0.699917（容差 ±0.00001，读取冻结产物应精确） |
| `AUC(alt)` | 0.697704（同上） |
| `AUC(max2)` | **0.701275**（同上） |
| `Spearman(main, alt)` | **0.94755** |
| test 行数 | 6398 |

第三臂必须同时满足：

```python
accepted = (
    te_auc > 0.69700
    and corr_main < 0.90000
    and gain > 0.00100
)
```

即：

| 门槛 | 必须达到 |
|---|---:|
| `AUC(main_te)` | `> 0.69700` |
| `Spearman(main_te, main)` | `< 0.90000` |
| `AUC(max3) - AUC(max2)` | `> 0.00100` |
| 因此 `AUC(max3)` | `> 0.702275` |

三项任一失败：

- 不生成 SUPER714 submission；
- 不改 bins/smoothing/超参补救；
- 结论写为“TE 被现有 CTR 吸收或强度不足”；
- 继续使用线上 best v1。

## 7. 通过后的提交约定

```python
submission = submit_sample[["id"]].copy()
submission["label"] = np.clip(max3_test, 0.001, 0.999)
```

检查 ID 顺序与 `test.csv`、`submit_sample.csv` 完全相同，label 有限且在 `[0.001, 0.999]`。文件名可用 `submission_super714_te_max3.csv`。

不要用 test 预测分布、test 赢率或公开榜反馈决定是否准入；唯一准入依据是第 6 节三项 OOF 门槛。

## 8. 结果预期

历史同类“加一个臂”的 OOF→LB 迁移约 37.6%～39.6%。刚过 `+0.001` 门槛时，中心预期 LB 约 **0.71491～0.71493**，有机会超过 0.71453，但没有保证。若完整 8seed 结果不过门槛，应停止；旧臂、伪标签、非 CB 和 broad TE 都已有负证据。

STATUS: READY_FOR_ML
