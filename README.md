# 车险索赔预测 — V4（当前交付）

> **提交文件：`submissions/submission_v4.csv`**  
> 诚实嵌套 OOF（20 block seed 均值）：**0.70303**（详见 [`docs/V4.md`](docs/V4.md)）  
> 对照 V3：同口径嵌套均值 0.70124 → **+0.00179**  
> 目标 0.707：**未达成**；诚实性门全部通过  
> V3 只读对照：`submissions/submission_v3.csv` / [`docs/V3.md`](docs/V3.md)

赛题：根据保单与车辆等多维信息，预测投保人未来一年内是否发生索赔。评价指标 ROC-AUC。  
数据在 `data/`（`train.csv` 14930 行含 `label`，`test.csv` 6398 行，`submit_sample.csv` 为提交模板）。

V4 在 V3 诚实协议上继续推高：不解冻早停、不碰测试标签、融合规则先登记再看分。  
有效增益来自 **更多折方案多样性（10-fold + 20-fold）** 与 **额外种子块进入 `max` 融合**，不是换模型族。

| 项 | 内容 |
|---|---|
| 提交文件 | `submissions/submission_v4.csv` |
| 主报口径 | 嵌套选择对 20 个 block seed（1000..1019）取均值 |
| 提交规则 | `views_max_10_20_r16_r16b` = max(rank) over 10-fold / 20-fold / r16 / alt_r16b 臂 |
| 对照 V3 | 嵌套 0.70124；V2 公开榜 0.70878（本地嵌套 0.69824） |
| 独立审核 | `src4/audit_v4.py`（与 `fuse4` 共用规则表） |

先读：

1. [`docs/V4.md`](docs/V4.md) — V4 数字、复现、无效方向、下一步建议  
2. [`docs/V4_SUPERVISION.md`](docs/V4_SUPERVISION.md) — 独立监督者定稿意见  
3. [`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md) — 数据生成机制（方案地基）  
4. [`docs/V3.md`](docs/V3.md) — V3 对照（只读）

---

## 复现

秒级（推荐，用仓库内已有臂）：

```bash
python3 -m pip install -r requirements.txt
bash run_v4.sh
```

等价手写：

```bash
PYTHONPATH=src2:src3:src4 python3 src4/fuse4.py \
  --dir artifacts/v4 --submission submissions/submission_v4.csv
PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 --submission submissions/submission_v4.csv --target 0.707
```

从零重训整条 V4 管道耗时长（多天 @ 4 核），步骤见 [`docs/V4.md`](docs/V4.md) §复现。  
复现 V3：`bash run_v3.sh`；复现 V2：`bash run_all.sh`。

---

## 目录

```text
data/                     比赛数据 + SHA256
run_v4.sh                 ★ V4 一键复算融合 + 监督（秒级）
run_v3.sh / run_all.sh    V3 / V2 重训（保留）

src4/                     ★ V4 融合 / 训练入口 / 监督者
src3/                     V3（只读对照）
src2/                     V2 特征工程（V3/V4 复用）

artifacts/v4/               ★ 最终臂预测 + fusion_report_v4.json
artifacts/audit_v4/        监督者报告
artifacts/v4_*_parts/      各实验 part（可断点续跑 / 重合并）

submissions/submission_v4.csv   ★ 当前交付
submissions/submission_v3.csv   V3 对照
submissions/submission_v2.csv   V2 对照（公开榜 0.70878）

docs/V4.md                V4 说明、复现、下一步、坑
docs/V4_SUPERVISION.md    监督者定稿
docs/DATA_STRUCTURE.md    数据地基
docs/V3.md / SUPERVISION.md / HANDOFF.md   历史对照
```

---

## 关键结论（跨 V2→V4）

1. **27/44 列是匿名化噪声**，对标签零预测力；但它们作为扰动编码仍对 CatBoost 有用。见 `DATA_STRUCTURE.md`。  
2. **`condition` 必须按车型归一化**；`days / cond_r` 是最强单列（AUC ≈ 0.620）。  
3. **抬分杠杆是「编码平均 + 每折多看数据 + 多样种子进 max」**，不是换 LGB/GLM/NN，也不是狂调超参。  
4. **诚实协议下 Bayes 天花板约 0.706**；本地 0.707 目标与「绝对不过拟合」目前不可兼得。见 `V4_SUPERVISION.md`。
