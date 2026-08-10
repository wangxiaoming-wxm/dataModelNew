# 独立审计：V4max3pro / `submissions/submission_v4max3pro.csv`

审计立场：独立审计员，不继承交付方叙事。全部数字由 `audits/audit_v4max3pro_part1_reproduce.py`
与 `audits/audit_v4max3pro_part2_optimism.py` 现场复算（只读，未重训、未改动任何既有文件）。
原始输出见 `audits/output_part1.txt`、`audits/output_part2.txt`。

## 裁决

**PROTOCOL_RISK** —— 未发现作弊或标签泄漏，声明数字全部可复算；但本地 +0.00215 含已知的
ES/10 折乐观与显著选择偏差，公开榜大概率低于 0.71437 的乐观外推。

## 一、无作弊 / 无泄漏（已逐项验证）

| 检查 | 结果 |
|---|---|
| `data/test.csv` 是否含 label | **不含**（44 列，无 label），结构上不可能看测试标签 |
| 数据是否被改动 | `sha256` 与 8/9 审计记录**完全一致** |
| 特征工程用 label | **没有**。`src2/features.py` 全程 label-free；`feature_blocks/*` 均 `uses_target=False` 且 `_without_targets()` 剔除 `label/target/y/id` |
| 目标编码（TE） | **全仓无 TE 实现**，仅有 `"target_encoding": "none"` 的声明 |
| 分箱是否见验证行标签 | 否。`train_noxb10.py` 走 `VIEWS["b5"]`，逐折 `fit_transform(X_tr)` → `transform(X_va/X_te)`；分位点只由训练折特征拟合 |
| test 是否进 `eval_set` | 否。全部 `eval_set` 均取自训练矩阵的验证折 |
| test transductive | 存在 `concat([train.drop(label), test])`，但**仅用于 label-free 分位点/尺度**（`fit_edges`），符合"仅限无标签统计" |
| OOF 行序对齐 | 各 npz 内存的 `y` 与 `train.csv` 标签逐行完全相同 |
| 打乱标签 sanity | 融合 OOF AUC = 0.5166（预登记带 0.47–0.53 内） |
| 提交文件格式 | 6398 行、id 与 `submit_sample.csv` 顺序一致、无 NaN/重复、范围 [0.001, 0.999] |

## 二、提交文件可复现（bit-exact）

- `max(rank(merger_ord8), rank(v2_cat_alt8), rank(ord_noxb_bag), rank(plus_strong), rank(noxb10))`
  再 `clip(0.001, 0.999)`：与 `submission_v4max3pro.csv` 的
  **spearman = 1.00000000，最大绝对误差 1.1e-16**。
- 基线锚定：冻结三臂重算的 test 与实测 0.71222 的 `submission_v4_max3.csv`
  **spearman = 1.00000000** —— 冻结臂确实就是那个真实榜分的来源。
- `noxb10.npz` 由 8 个 `part_noxb10_s*.npz` 重建，spearman = 1.0；
  `plus_strong.npz` 按文档口径（plus_v10 4 seed + Plain10 三个种子 rank 混合）重建 spearman = 0.99993。
- `status_report.json` 全部 8 个关键数字与我的复算差 < 5e-6。

## 三、风险证据（为什么不是 PASS_HONEST）

1. **delta 的置信区间穿过 0。** 配对 bootstrap（2000 次，按类分层重采样）：
   点估计 +0.002147，sd 0.001111，**CI95 = [-0.000057, +0.004301]**，P(delta≤0) = 2.8%。
   在只挑一次的前提下都仅约 1.9σ。

2. **选择偏差被低估。** 我按同一逻辑枚举 584 个"保留 ≥2 个 max3 核心臂"的候选配方：
   **66 个**通过预登记的 `delta ≥ +0.0015` 门槛，最终配方排第 **2** 名。
   门槛对 584 个高相关候选几乎没有筛选力，等于事后取最大值。

3. **noxb10 的本地增益基本不可迁移。** 加 noxb10 后：
   - 本地 nested +0.00082（占声称 +0.00215 的 38%）
   - 但与 max3 的 **test spearman = 0.99973**，只有 5.9% 的 test 行秩位移 >1 个百分位
   - OOF 侧位移是 test 侧的 **2.5 倍**（mean |rank shift| 0.00819 vs 0.00327）
   - 与已冻结的 `ord_noxb_bag` **test spearman = 0.9988**（同一 B5-noxb 世界，仅 5 折→10 折）

   即：它几乎不改变提交内容，却在本地记账里拿到 38% 的增益。

4. **ES 臂在 max 中系统性占便宜。** 五臂在 max 中的"夺冠占比"，OOF 减 test：
   `noxb10 +0.0298`、`ord_noxb_bag +0.0233`（两只 ES 臂全部上升）；
   `v2_cat_alt8 -0.0332`、`plus_strong -0.0165`（非 ES 臂全部下降）。
   方向系统一致，正是 ES 早停造成的 OOF 秩膨胀。

5. **真正新增的训练量几乎没换来东西。** 全部用旧库存的 `max3 + plus_v10` = 0.704293（+0.00122）；
   本轮新训后 `max3 + plus_strong` = 0.704533（+0.00146）。
   本轮数小时新训练的净贡献 ≈ **+0.00024**（远低于 sd 0.00111），其余靠 noxb10 的不可迁移增益。

6. **plus_strong 未达方案自己的预登记门槛。** `V4MAX3PRO_PLAN.md` P1 要求"单臂 ≥0.690"，
   实测 `plus_strong` OOF = **0.68911**，未达标却仍进入最终配方。
   （附带：剔除坏种子 2033 这件事**不构成**乐观来源——保留 2033 的 plus 混合 OOF 同为 0.68910。）

7. **`max` 规则的本地优势随臂数放大，且 k=5 无榜面验证。**
   `max` 相对 rank-mean：k=3 时 +0.00243，k=5 时 **+0.00503**。
   只有 k=3 的 max 被 0.71222 实测验证过。

8. **可复现性缺口（非作弊，但影响可核查性）。**
   `artifacts/v4max3pro/fuse_report.json` 是 01:54 的旧产物，其中 `admit=true`、
   `chosen` 是纯库存的 `...+ordered_bag+b7_closest`；而现行 `src4/fuse_v4max3pro.py`
   的 `has_new_train` 判据会拒绝该配方。实际提交文件（06:44）不是这份报告写出来的，
   `plus_strong.npz` 也没有已提交的构建脚本（`build_plus8.py` 写的是 `plus_v10_8.npz`）。
   我是靠 committed **产物**而非 committed **代码**才做到 bit-exact 复现。

## 四、口径没有作假的地方（应当记功）

- 未改考评折数刷分：nested delta 在 blocks = 1/2/3/5/8/10/20 下分别为
  +0.00236/+0.00233/+0.00231/+0.00215/+0.00218/+0.00218/+0.00218，**对尺子不敏感**。
- 没有取全局最优：真正的第 1 名是含 `b7_closest` 的 0.705430，交付方主动放弃了这 +0.0002，
  理由是 b7_closest 已知乐观 —— 这是对自己不利的正确选择。
- 文档没有夸大：`docs/V4MAX3PRO.md` 首句即写"不能诚实声称会到 0.7155"，
  风险段明确点出 ES/10 折 OOF 乐观与 plus 家族可能压缩 CV→LB 间隙，并建议不交。
  叙事与证据一致。

## 五、对最后一次提交机会的独立意见

**不建议交。**

- 目标 0.7155 按同口径需要 nested ≈ 0.706353，实际 0.705220，**差 +0.00113**，证据门槛未达。
- 扣掉 noxb10 不可迁移的 +0.00082 后，有效增益约 +0.00146（且其中 +0.00122 用旧库存即可拿到），
  对应现实 LB 约 **0.7130–0.7140**，中位期望比 0.71222 只高约 +0.001。
- 下行风险不对称：plus 来自 B7 家族（历史 CV→LB 间隙仅约 +0.0045，远小于 max3 的 +0.00915），
  若间隙被压缩，这一版可能**低于**已知的 0.71222，而 0.71222 是已经拿到手的确定分。
- 用最后一个名额去赌一个 CI 穿零、且 38% 增益来自几乎不改变提交内容的臂的 +0.001，
  风险收益比不划算。留名额优于交。
