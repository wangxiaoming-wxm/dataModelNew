# 车险索赔预测 — V3（本分支当前交付）

> **提交文件：`submissions/submission_v3.csv`**  
> 诚实嵌套 OOF（20 block seed 均值）：**0.70124**（详见 [`docs/V3.md`](docs/V3.md)）  
> 对照 V2（公开榜 0.70878）：同口径嵌套均值 0.69824 → **+0.003**

赛题：根据保单与车辆等多维信息，预测投保人未来一年内是否发生索赔。评价指标 ROC-AUC。
数据在 `data/`（`train.csv` 14930 行含 `label`，`test.csv` 6398 行，`submit_sample.csv` 为提交模板）。

本分支在 V2 的诚实协议上做迭代：不解冻早停、不碰测试标签、融合规则仍用预登记的
`views_max`，唯一有效的一步是把三个主臂改为 **10 折 CV**。

| 项 | 内容 |
|---|---|
| 提交文件 | `submissions/submission_v3.csv` |
| 本地口径 | 分层 10 折 × 8 种子，固定树数，无验证折早停；嵌套选择对 20 个 block seed 取均值 |
| 融合规则 | `views_max` = max(rank(cat_d5), rank(cat_d6), rank(cat_alt))（与 V2 相同） |
| 对照 V2 | `submissions/submission_v2.csv`，公开榜 0.70878，本地嵌套均值 0.69824 |
| 独立审核 | `src3/audit.py`、一次提交结论见 `artifacts/audit/oneshot_v2_vs_v3.json` |

先读：

1. [`docs/V3.md`](docs/V3.md) — V3 数字、复现、与 V2 的对照  
2. [`docs/SUPERVISION.md`](docs/SUPERVISION.md) — 天花板论证与监督者硬门  
3. [`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md) — 数据生成机制（方案地基）

---

## 复现

不重训、用仓库内已有臂预测复算融合与审核（秒级）：

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src2:src3 python3 src3/fuse2.py --dir artifacts/v3 --submission submissions/submission_v3.csv
PYTHONPATH=src2:src3 python3 src3/audit.py --dir artifacts/v3 --submission submissions/submission_v3.csv
```

从零重训 V3（4 核约 8–10 小时）：

```bash
bash run_v3.sh
```

复现 V2（旧口径，约 3.5 小时）：

```bash
bash run_all.sh
```

---

## 目录

```text
data/                  比赛数据 + SHA256
run_v3.sh              ★ V3 一键从零重训到提交
run_all.sh             V2 一键重训（保留）

src3/                  ★ V3 训练 / 融合 / 监督者
src2/                  V2 方案（未改动，V3 复用其特征工程）

artifacts/v3/             ★ V3 最终臂预测 + fusion_report_v3.json
artifacts/v3_f10/        10 折合并后的主臂
artifacts/worlds10/      每种子 part（可断点续跑）
artifacts/v2/             V2 最终产物（对照）
artifacts/audit/          监督者与一次提交审核

submissions/
  submission_v3.csv    ★ 当前交付
  submission_v2.csv    V2（公开榜 0.70878）

docs/
  V3.md                V3 说明与复现
  SUPERVISION.md       独立监督者结论
  DATA_STRUCTURE.md    数据地基
  RESULTS.md / HANDOFF.md / EXPERIMENTS.md   V2 文档（仍有效）

hunt/                  泄漏排查、天花板、一次提交审核脚本
logs/worlds10/         V3 训练原始日志
```

---

## 关键发现（继承自 V2）

**1. 44 个特征里有 27 列是匿名化噪声，对标签的预测力严格为零。**  
细节见 [`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md)。

**2. `condition` 必须先按车型归一化；`days / cond_r` 是全数据最强单列（AUC 0.620）。**

**3. 真正抬分的杠杆是「对编码做平均」和「每折多看训练数据」，不是换模型族、也不是调超参。**  
超参见屏全部落在噪声带；新编码世界 w4/w5 解耦但偏弱，未进最终规则。

**4. 诚实协议下 Bayes 天花板约 0.703；本地 0.99999 与「绝对不过拟合」不可兼得。**  
见 [`docs/SUPERVISION.md`](docs/SUPERVISION.md)。
