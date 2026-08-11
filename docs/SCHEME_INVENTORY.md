# SUPER714 历史方案台账

## 1. 审计结论与口径

本台账以“真实线上回执 > 可复算产物 > 代码/报告声明”为证据优先级。最终线上冠军固定为：

| 方案 | 本地 | 真实 LB | 结论 |
|---|---:|---:|---|
| `real714/v1_best714/src/explore_best.py` | pooled OOF **0.701275** | **0.71453** | 当前唯一提交基座 |

旧策略和特征文档中的 `0.71464` 与可复现包、代码日志及任务硬事实冲突，视为过期误记，不参与校准。

本地数字有 pooled、full、连续 5 块、20-block、真嵌套及含 ES 乐观值等不同口径，不能横向直接排序。下表保留原方案口径；“—”表示没有可靠记录，而不是 0。

## 2. ZIP 与材料来源

| 来源 | SHA-256 | 内容/核验 | 采用结论 |
|---|---|---|---|
| `/workspace/real714.zip` | `ed4cf82bf2ee8dd8113877c23986b95a374b1faae92c4746c7bd64ee9d4d1b9f` | best v1、opus5 max2、OOF/test 产物、提交 | 最高优先级 |
| `/tmp/schemes/714.zip` | `346a63bd6bc6c8c521ae30273d945e3d95043ff2a379ebfbbb30cdf189700f96` | 最新 `explore_best.py` 与旧 D_online | best 算法与 real714 一致；real714 是整理后的可复现版本 |
| `/tmp/schemes/715.zip` | `178b216509f27c3cf534f2319141ab19e61ab60337c3a55e1088075e31db496f` | 仅 D_online 与 `feat_semantic.py` | 两文件分别与 714 旧代码逐字节相同，不是新方案；且历史复盘已标记 715 不可信 |
| `/tmp/schemes/opus5.zip` | `e57b1041e2345a096c33aeaf400459a5c4cf8f9a367f101513686a4febf89552` | v4_honest、v4_max3、v5_honest | 用于三条真实 LB 锚点与臂产物 |
| `/tmp/schemes/zcode.zip` | `200f0d0fe923a25c327e1420214139991111a779994534327ba9ed86992eb120` | Phase1～v6、日志、OOF、提交 | 用于消融、ES 偏差和正交性证据 |
| `/tmp/schemes/subs` | — | 8 个历史提交；均与 best v1 高相关 | 用于提交端相关性审计 |
| `/tmp/schemes/arms`、`arms2` | — | v2/B7/zcode 的可用 OOF 与 test 预测 | 用于单臂和 max3 探针 |

## 3. CreateDataModel / real714 系列

### 3.1 D 特征体系与 RMSE 路线

| 方案 | 本地 OOF | 真实 LB | 核心变化 | 结论 |
|---|---:|---:|---|---|
| v21/benchmark | 0.69270 | 0.70168 | CB，5seed×5fold×3bag，Top-20 交叉 | 早期基线 |
| v28 TE | 0.75469 | 0.69087 | D 帧直接加入 8 路平滑 TE | 严重虚高，否决 |
| v30 | 0.68687 | — | 移除 code/grades 等 | 反降 |
| v31 | 0.69360 | 0.70251 | Top-22 交叉 | 小幅有效 |
| v32 | 0.69315 | — | depth7/1500/bagging | 调参反降 |
| v33 | 0.69491 | 0.70324 | Top-22 + 16 个比值 | 有效但远弱于 opus5 |
| v34 | 0.69522 | 0.70214 | 49 个连续变换×类别交叉 | 本地升、线上降 |
| D_reg | 0.69437 | — | v1/Top22 + RMSE + 3bag | RMSE 方向有效 |
| D_online | 0.69597 | 0.70457 | v33 + RMSE + 10bag | D 路线上限 |
| D_tune | — | — | depth/lr/iter/bag 扫描 | 边际耗尽 |
| D_x20 | 0.69584 | — | x20 残差 5 特征 | `-0.00013`，否决 |

### 3.2 opus5 / best 系列

| 方案 | 本地 | 真实 LB | 核心变化 | 结论 |
|---|---:|---:|---|---|
| opus5 原版 / v4_honest | nested 0.69993 | **0.71104** | Logloss、双编码世界、max2 | best v1 的直接前身 |
| **best v1** | pooled **0.70128** | **0.71453** | RMSE、8seed×3bag、Ordered main + Plain alt、max2 | 当前冠军 |
| best v2 | max2 0.70069；max3 0.69917 | — | 第三 z-score 世界、12seed | 弱第三世界拖累 |
| best v3 | 0.70037 | — | alt 改 Ordered+d5 | 臂变同质，融合反降 |
| best v4 | — | — | alt 的 l2/iter/depth 5 组扫描 | 全负 |
| best v5 | max2 0.70128；max3 0.69933 | — | 三臂同步重跑 | 复核 v1，第三臂仍失败 |
| best v6 | 0.70152 | 线上反降，未留精确值 | main 加 5 个新比值 | 本地微升不迁移 |
| best v7 | 0.70125 | — | main l2=3 | 反降 |

有效增益只有四项：`cond_r/ratio`、RMSE、3bagging、两种编码世界的 rank-max2。扩 seed、同族调参、第三 z-score 世界和新比值均已证伪。

## 4. 全部真实提交锚点

| 文件/方案 | 本地（原口径） | 真实 LB | 结论 |
|---|---:|---:|---|
| submission_0807 | 候选 0.67624，映射未完全确证 | 0.68749 | 早期基线 |
| submission_v10 | nested 0.701315 | 0.70570 | B5+plus 初代 |
| b7_closest_honest | 0.702705 | 0.70722 | B7 家族上限 |
| submission_b6pro | 无可靠记录 | 0.70208 | 弱于 B7 |
| codexdp_v1 | 无可靠记录 | 0.70722 | 与 B7 持平 |
| zcode b7pro / submission_best（旧） | 0.702888 | 0.70710 | 高相关五臂无突破 |
| submission_v2 | nested 0.69856 | 0.70878 | `cond_r/ratio` 路线起点 |
| submission_v3_max3 | 修正 nested **0.69870** | **0.71184** | 原 0.702264 报告已纠正 |
| submission_v3 | 10 折 nested 0.70124 | 0.71064 | 本地增量仅部分迁移 |
| submission_v5 | — | 0.71035 | 弱编码世界 |
| submission_v4_max3 | 0.70307，含 ES | **0.71222** | 旧体系冠军，但低于 best v1 |
| submission_v5_honest | 0.70253 | **0.71207** | 纯诚实第三 gap 臂，增益真实但不足 |
| submission_v4_honest | 0.69993 | **0.71104** | 纯诚实 max2 |
| submission_0811 | 无记录 | 0.67982 | 来源不明异常，剔除 |
| submission_v4ext | 真嵌套 0.70381 | **0.71123** | 13 个高相关臂负迁移 |
| **real714 best v1** | pooled **0.70128** | **0.71453** | 总冠军 |

任务给定的五条对照 LB（v4_max3、v5_honest、v3_max3、v4ext、v4_honest）均与产物/复盘一致。

## 5. zcode / opus5 内部实验与未提交库存

| 阶段/方案 | 本地结果 | 线上 | 结论 |
|---|---:|---:|---|
| B5 真实 8seed | pooled 0.698175 | — | 干净早期单臂 |
| Phase1 final_d7xbin | 0.6980 | — | xbin 有效但同质 |
| Phase1 blend3 | 0.6986 | — | 概率融合边际 |
| Phase1 TE arm | 0.62～0.65 | — | 广谱 TE 弱；旧实现训练行还看到自身标签 |
| Ordered 单 seed | 0.691639 | — | 唯一胜过 control 的训练机制 |
| ordered_bag | 0.699107，含 ES | — | 与 Plain 有一定互补 |
| ord_noxb_bag | 0.700640，含 ES | — | ES 乐观约 `+0.0015` |
| Phase2 2-arm | full 0.701445；LFO 0.701556 | — | 稳健但旧体系内无 LB 优势证据 |
| merger_ord8 | 0.696596 | — | 诚实 main/Ordered |
| v2_cat_alt8 | 0.697039 | — | 诚实 alt |
| mine_noxb8 honest | 0.694988 | — | 太弱，进 max 掉分 |
| alt-world Ordered | 0.691658 | — | 第三编码世界失败 |
| v4 max4 | 0.70321，含 ES | 未提交 | 比 max3 仅 `+0.00014`，不值得加臂 |
| v6_zcode | 0.70326，含 ES | 未提交 | 与旧冠军同族，预期仍在 0.712 噪声带 |
| v4max3pro | nested 0.70522，含 ES/10 折 | 未提交 | plus 弱、noxb10 孪生，否决 |
| v4max3pronew | nested 0.70557，ES 污染 | 未提交 | 外部变体，不采纳 |
| codexv810 | m20 0.70536；nested5 0.70491 | 未提交 | B7/高相关结构风险 |
| V4 main | 真嵌套 0.70303 | 未提交 | 被 v4ext 本地支配，且同族 |

## 6. 已证伪方向总账

1. **同族堆臂**：v4ext 13 臂平均相关性 0.973，OOF 选择膨胀约 0.0028，LB 反降到 0.71123。
2. **广谱 TE 直接塞入旧 D 帧**：v28 `0.75469 → 0.69087`；旧 LGB-TE OOF 0.67110。
3. **弱第三编码世界**：best v2/v5、alt2、alt-world Ordered 全部使 max3/max4 下降。
4. **伪标签**：v27 约 0.65，禁止。
5. **非 CatBoost 家族**：LGB、XGB、EBM、GLM、NN 均弱且没有可迁移正交性。
6. **监督 stacking / 权重搜索**：LR stack 约 0.7003，低于无监督 max；存在选择过拟合。
7. **无效模型变化**：Lossguide 0.679512、MVS 0.6899、深树/l2/扩 seed 均无稳定增益。
8. **无效特征扩展**：全配对交叉、周期 days、匿名噪声统计、x20 残差、新比值、分群建模均无真实 LB 证据。

## 7. 唯一保留的待验证方向

仅保留“**单一、预注册、折内双层诚实的 `TE(source×days_bin10)` CatBoost 臂**”。它与 v28/旧 TE 的区别是：

- 只编码一个有业务含义的交叉，不扫描多列；
- 外层训练行使用内层 OOF TE，不允许训练行看到自身标签；
- 验证/test 只用外层训练折统计；
- 作为独立第三臂参加预注册 max3，不直接改写已验证的 best v1 两臂；
- 必须通过 OOF、相关性和融合增益三道门槛，任何一项失败即回退 best v1。
