# real714 — 可复现最佳方案包

从 `CreateDataModel` 大量实验版本中，按**已验证线上 AUC** 筛出最好的两版，整理为可独立复现的干净包。

| 优先级 | 目录 | 本地 OOF | 线上 AUC | 一句话 |
|:---:|---|---:|---:|---|
| **1（推荐提交）** | [`v1_best714/`](v1_best714/) | **0.70128** | **0.71453** | opus5 双编码世界 + RMSE + 3bagging + max2(rank) |
| 2（同源基线） | [`v2_opus5_711/`](v2_opus5_711/) | 0.69993 | **≈0.711** | 同特征族、Logloss、无 bagging 的诚实 max2 |

> 数据不在本包内。默认读取 `/Volumes/pssd/app/ml/正式比赛/data/{train,test}.csv`，可用环境变量 `DATA_DIR` 覆盖。

---

## 为什么选这两版（不是 v2～v8 / D方案）

依据 `CreateDataModel/版本实验记录.md`：

- **best v1（explore_best.py）** 是第三阶段唯一确认的线上最优：**0.71453**。
- **opus5 诚实 max2** 是其直接前身（组内线上约 **0.711**），特征体系相同、训练更“朴素”，适合对照与快速 fuse 复现。
- **best v2～v7** 本地无增益或线上反降，已全部证伪。
- **D_online** 线上 0.70457，低于上述两版，不放入本包。
- **v21/benchmark** 线上 0.70168，更早的纯 CB 基线，不放入本包。

---

## 快速开始

```bash
# 依赖
pip install -r requirements.txt

# --- 方案1：验证已保存产物（秒级）---
python3 v1_best714/verify_artifacts.py
# 预期: fuse OOF ≈ 0.70128

# --- 方案1：从数据完整重训（约 1.5～2.5 小时，视 CPU）---
bash v1_best714/reproduce.sh
# 可选冒烟: bash v1_best714/reproduce.sh --smoke

# --- 方案2：用预计算臂秒级融合 ---
python3 v2_opus5_711/src/fuse_v4b.py
# 预期 nested ≈ 0.69993

# --- 方案2：完整重训（约 150 min）---
bash v2_opus5_711/reproduce.sh
```

自定义数据路径：

```bash
export DATA_DIR=/path/to/dir   # 内含 train.csv / test.csv
bash v1_best714/reproduce.sh
```

---

## 目录结构

```
real714/
├── README.md                 # 本文件
├── requirements.txt
├── docs/
│   └── 版本实验记录.md       # 从 CreateDataModel 复制的台账
├── v1_best714/               # ★ 线上 0.71453
│   ├── README.md
│   ├── reproduce.sh
│   ├── verify_artifacts.py
│   ├── src/explore_best.py
│   ├── docs/特征设计文档_best_v1.md
│   ├── artifacts/{best_oof,best_test}.npy
│   └── submissions/submission_best.csv
└── v2_opus5_711/             # 同源诚实基线 ~0.711
    ├── README.md
    ├── reproduce.sh
    ├── src/ ...
    ├── artifacts/*.npz
    └── submissions/submission_v4_honest.csv
```

---

## 复现验收标准

| 检查项 | v1_best714 | v2_opus5_711 |
|---|---|---|
| 本地融合 OOF | max2 ≈ **0.70128**（±0.0003） | nested ≈ **0.69993** |
| 提交行数 | 6398 | 6398 |
| 列 | `id,label`，label∈[0.001,0.999] | 同左 |

随机性已由固定 seed / fold / bagging 种子锁定；CatBoost 多线程下可能有极小浮点差，不应改变到 0.001 量级。

---

## 来源说明

- 源仓库实验场：`/Volumes/pssd/app/ml/正式比赛/CreateDataModel`
- v1 源文件：`explore_best.py` + `特征设计文档_best_v1.md` + 已跑通产物
- v2 源目录：`CreateDataModel/20260810-curos-opus5-0.711/`（原样整理，去掉 `._*`）
