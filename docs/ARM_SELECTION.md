# SUPER714 臂选择与融合结论

## 1. 结论

推荐的 SUPER714 是**有条件的三臂方案**：

1. best v1 主臂 `main`：cond_r 世界；
2. best v1 次臂 `alt`：rate 世界；
3. 唯一候选新臂 `main_te`：主臂特征帧 + 折内双层诚实 `TE(source×days_bin10)`。

融合固定为三个 rank-pool 臂的逐行 `max`。第三臂只有同时通过三条预注册门槛才准入；否则 SUPER714 判定失败，保留原 best v1，不生成“新方案”提交。现有证据支持它是唯一值得验证的方向，但不能在第三臂尚未完整训练时承诺一定超过 LB 0.71453。

## 2. best v1 基座复算

来源：`real714_pkg/v1_best714/artifacts/best_oof.npy`，标签来自原始 train。

| 臂 | pooled OOF AUC | 设定 |
|---|---:|---|
| `main` | **0.699917** | RMSE，Ordered，d5，800 iter，l2=10 |
| `alt` | **0.697704** | RMSE，Plain，d6，800 iter，l2=6 |
| `max(main, alt)` | **0.701275** | 当前 best v1 |

`main` 与 `alt`：Pearson 0.94803，Spearman **0.94755**。两臂强度接近但编码世界不同，max2 相对最强单臂增加 0.001358，并已在线上兑现为 0.71453。

## 3. 历史提交端相关性

以下均与 best v1 的 `submission_best.csv` 比较，使用 6398 行 `label` 的 Spearman 相关性：

| 历史提交 | Spearman(best v1) | 已知 LB/状态 | 判定 |
|---|---:|---:|---|
| `beat_max3` | 0.97756 | 来源/LB 未确证 | 不作为证据 |
| v2 | 0.98646 | 0.70878 | 同信号轴 |
| v3 | 0.98923 | 0.71064 | 同信号轴 |
| v3_max3 | 0.98586 | 0.71184 | 同信号轴 |
| v4 | 0.99004 | 未确证 | 同信号轴 |
| v4_max3 | 0.98629 | 0.71222 | 低于 best v1 |
| v5_honest | 0.98660 | 0.71207 | 低于 best v1 |
| v4_honest | 0.98876 | 0.71104 | 低于 best v1 |

所有可用历史提交与 best v1 的相关性都高于 0.977。历史库存没有一个“已证明强且提交端正交”的现成候选，不能靠旧提交再做一次 max。

## 4. 可用 OOF 的第三臂 max 探针

探针基线为 best v1 pooled `max2=0.701275`。对旧产物先做全局 rank，再计算 `max(best_main, best_alt, old_arm)`。由于不同产物的 fold、损失和 ES 协议不同，下表只用于排除，不可当作新方案无偏 OOF。

| 旧产物 | 单产物 OOF | Spearman 对 best main | max3 Δ | 否决原因 |
|---|---:|---:|---:|---|
| B7 closest（本身已是融合） | 0.702705 | 0.9577 | +0.004697 | 不是单臂；对应家族 LB 仅 0.70722 |
| v10/B7 fuse | 0.702209 | 0.9564 | +0.004533 | 本身为融合；跨协议 max 选择偏差 |
| gap_bag | 0.698906 | 0.9656 | +0.003346 | 高相关；旧 LB 未超 0.71207 |
| ord_noxb_bag | 0.700640 | 0.9662 | +0.003325 | ES 乐观；旧 v4_max3 LB 0.71222 |
| xbin_bag | 0.696719 | 0.9595 | +0.001488 | 高相关且偏弱 |
| plus | 0.688617 | 0.9256 | +0.000169 | 未过强度和增益门槛 |
| cat_d5 | 0.695064 | 0.9802 | +0.000072 | 孪生臂 |
| old gap | 0.690438 | 0.9588 | -0.000028 | 弱臂 |
| v2_cat_alt8 | 0.697039 | 0.9470 | -0.000409 | best alt 的旧同族版本 |
| merger_ord8 | 0.696596 | 0.9859 | -0.000510 | best main 的旧同族版本 |
| old cat_alt | 0.694070 | 0.9378 | -0.000771 | 弱 alt |
| LGB-TE | 0.671096 | 0.9212 | -0.003759 | 非 CB、弱 TE、明确死路 |

高 OOF 增量集中在“已经融合过的产物”或 ES/高相关旧臂上。它们真实 LB 全部低于 best v1，正是不能把跨协议 max 探针误当成可迁移增益的反例。

## 5. 为什么只保留新的单交叉 TE

### 支持证据

策略文档记录了两个独立轻量实验：

| 特征帧 | base | +TE | 增量 | 与旧 main 相关性 |
|---|---:|---:|---:|---:|
| 7 列简化帧 | 0.66277 | 0.70414 | +0.04137 | 0.8222 |
| 29 列扩展帧 | 0.64739 | 0.65615 | +0.00876 | 0.7509 |

这两组没有随仓库交付可复算 OOF，故只算**方向证据**，不能代替最终门槛。它们至少说明显式 `source×days` 历史索赔率可能形成区别于连续 `ratio` 的离散概率视角。

### 必须保留的反证

best v1 并非完全没见过该交互：

- main 已有类别交叉 `d10s = days_q10 × source`；
- alt 已有 `Ad13s = days_q13 × source`；
- CatBoost 会对类别交叉形成 ordered statistics。

因此新意仅在于“**外层训练折统计得到的单一显式平滑数值 TE**”，不是一个全新的原始交互。它可能被 CatBoost 原生 CTR 完全吸收，最终与 main 高度相关。这也是 `corr<0.90` 和 `max3 Δ>0.001` 不能放宽的原因。

旧 TE 的失败也不能忽略：

- v28：本地 0.75469、LB 0.69087；
- LGB-TE：OOF 0.67110，加入当前 max2 反降 0.003759；
- zcode 旧 TE：约 0.62～0.65；
- 旧 `te_arm.py` 给外层训练行直接映射包含自身标签的统计，不是双层诚实训练特征。

SUPER714 只验证一个预注册交叉，且训练行必须使用内层 OOF 编码，避免重走 v28。

## 6. SUPER714 最终配方

### 臂 A：`best_main`

- 特征：原 `explore_best.build_main`，不新增特征；
- CatBoostRegressor/RMSE；
- Ordered，depth=5，iterations=800，learning_rate=0.03；
- l2_leaf_reg=10，random_strength=0.7；
- 5-fold；seeds 2026～2033；每折 bag seeds 0/1/2；
- 固定树数，无 early stopping。

### 臂 B：`best_alt`

- 特征：原 `explore_best.build_alt`；
- CatBoostRegressor/RMSE；
- Plain，depth=6，iterations=800，learning_rate=0.03；
- l2_leaf_reg=6，random_strength=0.7；
- 其余 fold/seed/bag 与臂 A 相同；
- 固定树数，无 early stopping。

### 臂 C：`main_te`（唯一新训练臂）

- 基础帧：与臂 A 相同；
- 只增加数值列 `te_source_days_bin10`；
- days 分 10 个外层训练折等频桶；
- key：`source + "|" + days_bin`；
- smoothing=20.0；
- 外层 5-fold；每个外层训练折内部 4-fold OOF TE；
- 未见 key 回退到对应训练统计的正例先验；
- 模型超参、8 seeds、3 bagging 与臂 A 完全相同，以便把变化隔离到 TE；
- 禁止额外 TE 列、q5/q20 扫描、早停或参数搜索。

实现位于 `src_super/features_te.py`。

### 融合规则

每个 seed 的完整 OOF 先 rank；每个 seed 的 test（先平均该 seed 的 5 fold×3 bag）再 rank；8 个 seed 等权平均得到臂级 rank-pool。沿用 best v1 约定，不对臂级 pool 再拟合权重：

```python
max2_oof = np.maximum(best_main_oof, best_alt_oof)
max3_oof = np.maximum.reduce([best_main_oof, best_alt_oof, main_te_oof])

max2_test = np.maximum(best_main_test, best_alt_test)
max3_test = np.maximum.reduce([best_main_test, best_alt_test, main_te_test])
```

最终提交只对 `max3_test` 做 `[0.001, 0.999]` clip，不做监督 stacking、校准或榜单反馈调权。

## 7. 入场门槛

三项必须在同一套 14930 行 pooled OOF 上同时满足：

1. `AUC(main_te_oof) > 0.69700`；
2. `Spearman(main_te_oof, best_main_oof) < 0.90000`；
3. `AUC(max3_oof) - AUC(max2_oof) > 0.00100`。

固定锚点：

- `AUC(best_main_oof) = 0.699917`；
- `AUC(best_alt_oof) = 0.697704`；
- `AUC(max2_oof) = 0.701275`。

连续 5 块、逐 seed、test 相关性和 bootstrap 可以作为稳定性报告，但不能替代上述门槛，也不能据此搜索新规则。

## 8. 预期与否决项

旧体系中两个可对照的加臂迁移率：

- v4_honest → v5_honest：OOF `+0.00260`，LB `+0.00103`，迁移约 39.6%；
- v4_honest → v4_max3：OOF `+0.00314`，LB `+0.00118`，迁移约 37.6%（含 ES）。

若 TE 臂刚好以 `+0.00100` 过门槛，按 0.38～0.40 的历史折扣，中心预期约为 best v1 `0.71453 + 0.00038～0.00040 = 0.71491～0.71493`。这是“有机会超越”的证据化预期，不是 LB 保证。

以下任一情况直接否决新提交：

- 三条门槛任一不满足；
- 训练 TE 使用自身标签、验证标签或测试标签；
- 为过门槛扫描 bins、smoothing、权重、融合子集；
- 追加 gap、ES、plus、v4ext 同族臂；
- 改用非 CatBoost、伪标签、外部数据或公开榜探测；
- 将 broad TE 直接塞回 D 方案；
- 以跨协议 OOF max 探针替代同协议完整验证。
