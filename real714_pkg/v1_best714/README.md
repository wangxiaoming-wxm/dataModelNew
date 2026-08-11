# best v1 — 线上 AUC 0.71453

单文件双臂方案：`src/explore_best.py`

| 项 | 值 |
|---|---|
| 本地 pooled OOF (max2) | **0.70128** |
| 线上 AUC | **0.71453** |
| 臂1 | cond_r 世界 · Ordered · depth=5 · iter=800 · l2=10 · 8seed×3bag · RMSE |
| 臂2 | rate 世界 · Plain · depth=6 · iter=800 · l2=6 · 8seed×3bag · RMSE |
| 融合 | `max(rank(oof1), rank(oof2))` |

特征细节见 [`docs/特征设计文档_best_v1.md`](docs/特征设计文档_best_v1.md)。

## 依赖

```bash
pip install -r ../requirements.txt
```

## 数据

默认：`/Volumes/pssd/app/ml/正式比赛/data/{train,test}.csv`

```bash
export DATA_DIR=/your/path   # 含 train.csv / test.csv
```

## 复现

```bash
# 秒级：核对已保存 OOF / 提交文件
python3 verify_artifacts.py

# 完整重训（约 90～150 min）
bash reproduce.sh

# 冒烟（2-fold / 1seed / 1bag，只验证通路）
bash reproduce.sh --smoke
```

产出：

- `submissions/submission_best.csv`
- `artifacts/best_oof.npy` / `best_test.npy`（dict: main/alt/fuse）
- `explore_best.log`

## 预期日志片段

```
臂1 pooled OOF = 0.69992
臂2 pooled OOF = 0.69770
max2融合 OOF = 0.70128
```
