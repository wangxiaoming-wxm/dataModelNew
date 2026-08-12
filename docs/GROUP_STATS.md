# 分组统计特征（Group-Stats）探测结论

## 方向

按分类键对数值列做 **fold-local** `mean/std/count`（及 `value - group_mean`），仅用训练折拟合，验证/测试只读映射 + 次数平滑——无标签泄漏。

## 协议

| 项 | 设定 |
|---|---|
| 键 | region, source, month, version, grades, age_range, code, livability, x19 + 交叉 rs/sa/ra/sm/rv |
| 值 | days, condition, cc, V, max_g, x0/1/5/10/16/17/18 |
| 聚合 | mean, std, count, dev（行值−组均值） |
| CV | Stratified 5-fold，统计量只在训练折上估计 |
| 对照锚点 | fp_v4 OOF **0.71640** |

产物：`artifacts/group_stats/probe_metrics.json`

## 结果（证据）

### 1) 单独成臂再与 fp_v4 融合 → **负增益**

| 臂 | 臂 OOF | vs fp Spearman | 与 fp 融合 nested |
|---|---:|---:|---:|
| CatBoost(全 GS) | 0.662 | 0.83 | **−0.003** |
| LogReg(core keys) | 0.618 | 0.67 | **−0.0005** |
| LogReg(x 聚焦) | 0.547 | 0.26 | **≈0 / 微负** |

所有 `nest_fp` **≤ fp_v4 nested(0.71643)**。

### 2) 折内挑选「最正交」GS 再混入 → **仍负**

k=5..40 的 nested 相对 fp 折均 **Δ ≈ −0.003 ~ −0.006**。

### 3) 主模型内注入（烟测）→ **几乎无增益**

同一套 main 特征，CatBoost Ordered d5，iter=300，1 seed：

| | OOF |
|---|---:|
| main only | 0.68814 |
| main + fold-local GS | 0.68864 |
| Δ | **+0.00050** |

与 fp_v4 再融合后二者均 **低于** 纯 fp_v4。主臂与 +GS 预测 Spearman **0.963**。

## 解释

`build_main` 已有大量 **days/condition 分箱 × region/source/age 交叉** 与频次特征；组内 days 偏差等强信号与现有世界高度共线（单变 Spearman 对 fp 常达 0.55+）。  
因此「经典 group-stats」在本数据上 **已被吸收**，不是尚未开发的富矿。

## 决策

- **不晋级**、不替换 `submission_champion.csv`（仍为 fp_v4）
- 不把 GS 臂写入提交融合
- 若未来重训主模型，GS 最多作可选烟测；门槛须 **主臂 OOF 明显 > 现 main 且融合 > fp_v4**
