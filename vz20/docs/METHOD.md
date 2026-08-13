# vz20 方法详解

## 0. 结论先行
在**全新 outer seed(90210)** 的诚实 held-out 协议下，把 rebuild V2 的 rich/freq 特征世界作为
新臂并入 vz19，相对 vz19 配方的 fold-mean 提升仅 **+0.00021**，远低于门禁 **+0.0015**；
甚至在 eval 上作弊调权重的 **oracle 上界也只有 +0.0002**。**门禁未过**。原因是两方案信号高度同源
（ratio_rich 与 vz19 arm1 的 Spearman≈0.985）。本文件记录方法、诚实证据与失败归因。

## 1. 为什么要重建一个统一的评估标尺
vz19 的 0.70355 是"池化 OOF"口径，rebuild 的 0.695 是"nested"口径，两者不可直接比较（见上一轮审核）。
且 vz19 没有 checkpoint。因此 vz20 从零搭一个**统一、配对**的 held-out harness，让 vz19 配方与
vz20 配方在**同一批外折、同一批臂缓存**下对比，唯一差异是 vz20 是否并入 rich/freq 臂。

## 2. 数据与特征世界
- Train 14930（正例 10.02%），Test 6398，id=16 hex。
- 特征世界（全部 label-free、折内拟合）：
  - vz19 世界：`main`(cond_r) / `alt`(rate)，含 ~50 手工类别交叉（`src/features.py`）。
  - rebuild 世界：`ratio_rich` / `rate_rich` / `ratio_freq` / `rate_freq`（`ref_rebuild/features.py`，
    含预注册分位×region/source/age 交互与 reliability/rare 旗标）。

## 3. 臂（8 个 + byte07）
见 `docs/PROTOCOL.md` 表。每臂 = CatBoost RMSE，4 seeds rank 平均，800 trees。
- A1/A2 = vz19 arm1/arm2；REF1/REF2 = 独立 seed 的 ref 管线（供 max2）；
- R1..R4 = rich/freq 的 ratio/rate 实现；
- BYTE07 = id byte0+byte7 折内 target encoding（vz19 已验证的唯一 id 弱信号）。

## 4. 诚实 held-out 协议（等价 nested）
1. 外层 `StratifiedKFold(5, shuffle, seed=90210)`，全新 seed，未参与任何调参。
2. 每折：所有统计只在 outer_train 拟合 → 应用 outer_valid/test；outer_valid 标签只计一次分。
3. 配方权重**全部预注册**（0.64/0.36、0.5/0.5、0.89/0.11），不在 outer-valid 搜索。
4. vz19 与 vz20 复用**完全相同**的 A1/A2/REF/BYTE07 缓存 → 配对比较，低方差。

## 5. 预注册配方（唯一裁决用 = vz20 主配方）
```
ref          = r(0.5 r(REF1) + 0.5 r(REF2))
byte07       = r(等权(byte0_TE, byte7_TE))
# vz19 基线:
my_cb        = r(0.64 r(A1) + 0.36 r(A2))
vz19         = 0.89 max(my_cb, ref) + 0.11 byte07
# vz20: 每个世界升级为 3 实现家族的 rank 平均
ratio_family = mean(r(A1), r(R1), r(R3))
rate_family  = mean(r(A2), r(R2), r(R4))
my_cb_v20    = r(0.64 r(ratio_family) + 0.36 r(rate_family))
vz20         = 0.89 max(my_cb_v20, ref) + 0.11 byte07
```
`r(x)=rankdata(x)/n`。最终提交由该配方在**全 train** 重拟合各臂后预测 test。

## 6. 结果（outer seed 90210, 5 折, 4 seeds）
| 配方 | fold-mean | 5 折 |
|---|---|---|
| vz19 | 0.69957 | 0.71734 / 0.67316 / 0.70464 / 0.69605 / 0.70666 |
| vz20（主） | 0.69978 | 0.71732 / 0.67244 / 0.70368 / 0.69677 / 0.70867 |
| lift | **+0.00021** | 仅 2/5 折不退化 |

组件分解（本 harness，配对）：
- my_cb 0.69889；ref 0.69813；**max2 lift = −0.00007（≈0！）**；**byte07 lift = +0.00075**。
- 诊断备选：vz20_4arm +0.00034；vz20_max3 +0.00041；MEGA 8-arm +0.00020。
- **oracle 上界（在 eval 上调权重，非法）= +0.00020 @ w=0.95**。

## 7. 失败归因（第一性原理）
1. **信号同源**：ratio_rich(R1) 与 vz19 arm1(A1) 的 Spearman≈0.985，rate 世界各实现也 0.92~0.98。
   rich/freq 只是 vz19 手工交叉的**近重复实现**，没有带来正交信息。
2. **max2 在新 seed 上失效**：vz19 声称 max2 +0.0018，本 harness 在干净 outer seed 上测得 **−0.00007**。
   佐证上一轮审核判断：max2 的 OOF 增益是"对同一份 OOF 重采样"的乐观，不是可泛化增益。
3. **byte07 是唯一真实正交增益(+0.00075)**，已被 vz19 吃满（w=0.11）；协议禁止扩展更多 id 字节。
4. 因此**可用正交信号已被 vz19 榨干**；任何诚实融合的天花板≈+0.0002，无法达到 +0.0015。

## 8. 置乱哨兵
R1(ratio_rich) 用打乱标签重训（3 折, 2 seeds）：fold-mean 0.5128 / pooled 0.5129 ∈ [0.48,0.52] → 无泄漏。
