# vz20 预注册协议（训练/评估前锁定）

本文件在跑 full 评估**之前**写定。阈值与配方一经写定，不因 outer-valid 结果修改。

## 0. 目标
构建 vz20，在诚实 held-out 下相对 vz19 配方 fold-mean 提升 ≥ **+0.0015**，且 ≥3/5 外折不退化；
置乱 AUC ∈ [0.48,0.52]；无 test 标签 / 无伪标签 / 无泄漏；权重预注册。

## 1. 验证协议（held-out，等价 nested）
- 外层 `StratifiedKFold(5, shuffle=True, random_state=90210)`。**90210 是全新 outer seed**，未用于 vz19(2040-2047) 或 rebuild(2026) 的任何调参，故 outer-valid 对本方案是干净的。
- 每个外折：所有特征统计（source 中位数/百分位、分位边界、频率、reliability、byte→rate 映射）**只在 outer_train 上拟合**，应用到 outer_valid 与 test。outer_valid 标签只用于最终计分。
- 每个臂 = CatBoost RMSE，`BAG_SEEDS=[11,12,13,14]`（ref 臂用 `[101,102,103,104]`）的 rank 平均。full 用 4 seeds、800 trees。
- 配方权重**全部预注册**（见 §3），不在 outer-valid 上做任何网格搜索。
- vz19 与 vz20 复用**完全相同**的 A1/A2/REF1/REF2/BYTE07 缓存预测；两者唯一差异是 vz20 额外把 rich/freq 臂并入 my_cb。因此比较是**配对的**（低方差）。

## 2. 臂（arms，全部 label-free 折内拟合）
| 臂 | 特征世界 | boosting | depth | l2 | rsm | 来源 |
|---|---|---|---|---|---|---|
| A1 | main (cond_r) | Ordered | 5 | 10 | 1.0 | vz19 arm1 |
| A2 | alt (rate) | Plain | 6 | 6 | 0.3 | vz19 arm2 |
| REF1 | main | Plain | 6 | 6 | 1.0 | 独立 ref 管线（换 boosting+seed） |
| REF2 | alt | Ordered | 5 | 10 | 1.0 | 独立 ref 管线 |
| R1 | ratio_rich | Ordered | 5 | 10 | 1.0 | rebuild V2（nested 证明 +0.006） |
| R2 | rate_rich | Plain | 6 | 6 | 1.0 | rebuild V2 |
| R3 | ratio_freq | Ordered | 5 | 10 | 1.0 | rebuild（reliability/rare 旗标） |
| R4 | rate_freq | Plain | 6 | 6 | 1.0 | rebuild |
| BYTE07 | id byte0+byte7 折内 TE | — | — | — | — | vz19（cross-half 已验证） |

`r(x)=rankdata(x)/n`。

## 3. 预注册配方
共享组件：
```
ref     = r( 0.5*r(REF1) + 0.5*r(REF2) )
byte07  = 折内 TE(byte0,byte7) 的等权 rank
```

**vz19 基线（忠实复现其配方）**：
```
my_cb   = r( 0.64*r(A1) + 0.36*r(A2) )
vz19    = 0.89 * max(my_cb, ref) + 0.11 * byte07
```

**vz20 主配方（PRIMARY，唯一用于门禁裁决）**：
把每个"世界"从单臂升级为多实现家族的 rank 平均（= 把 rich/freq 特征作为新臂）：
```
ratio_family = mean( r(A1), r(R1), r(R3) )     # cond_r 世界的 3 种实现
rate_family  = mean( r(A2), r(R2), r(R4) )     # rate 世界的 3 种实现
my_cb_v20    = r( 0.64*r(ratio_family) + 0.36*r(rate_family) )   # 沿用 vz19 的 0.64/0.36
vz20         = 0.89 * max(my_cb_v20, ref) + 0.11 * byte07
```
所有权重(0.64/0.36, 0.5/0.5, 0.89/0.11)均来自 vz19/rebuild 的既有取值，**不重新搜索**。

**诊断用备选（不参与门禁，仅记录）**：
- v20_4arm: ratio_family=mean(r(A1),r(R1)); rate_family=mean(r(A2),r(R2))（不含 freq）
- v20_max3: 0.89*max(my_cb_v20, ref, rich) + 0.11*byte07, rich=r(0.65 r(R1)+0.35 r(R2))

## 4. 门禁（缺一不可）
1. fold-mean(vz20) − fold-mean(vz19) ≥ **+0.0015**；
2. ≥3/5 外折 vz20 不低于 vz19（配对，允许极小退化容差 1e-6）；
3. 置乱标签哨兵：对 R1 臂在 3 折上用打乱的 y 重训，pooled/fold-mean ∈ [0.48,0.52]；
4. 最终提交由锁定配方在**全 train** 重拟合各臂后预测 test 生成；记录 SHA256。

## 5. 禁止事项
- 不在 outer-valid 上搜索任何权重/配方/符号；
- 不使用 test 标签、历史线上、伪标签；
- 不加深更多 id 字节（byte6/3 已证伪）；不碰 fp_v8/championship 0.77 泄漏旁路；
- 不因 final-inner 分数高就替换主配方。

## 6. 失败处理
若主配方未过 +0.0015：诚实记录实际 lift 与最接近方案，**不注水**，STATUS.md 标注"未过门禁/不建议替换 vz19"。
