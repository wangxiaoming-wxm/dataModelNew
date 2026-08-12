# Lossguide 复验结论

## 外部说法

有人报告 Lossguide（leaf-wise）相对 Depthwise **+0.00386**，疑似突破口。

## 证据

| 来源 | 结果 |
|---|---|
| 仓库历史 zcode PHASE2（`max_leaves=48`） | OOF **0.679512**，相对基线 **−0.0115**（大幅伤害） |
| 外部后续 5-fold 复验（LG-d6-l2=6-ml31） | **0.67394** vs Depthwise **0.67412**（Δ **−0.0002**，持平） |
| 对方对 3-fold screen +0.0039 的解释 | **噪声**；换 5-fold 消失 |

## 判决

**REJECT / 不晋级。**  
3-fold 上的 +0.0039 不可信；本任务上 Lossguide 不能稳定优于 Depthwise/Ordered，更不可能抬升当前 fp_v5（OOF 0.72880）融合。

## 行动

- 终止继续扫 Lossguide 超参（单配置约 12 分钟，期望增益≈0）
- 转向其它未测方向（如特征剪枝）
