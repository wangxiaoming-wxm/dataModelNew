# 复现指南

## 环境

- Python ≥ 3.10（开发机为 3.12）
- 依赖见仓库根目录 `requirements.txt`

```bash
python3 -m pip install -r requirements.txt
```

如遇 pandas/numpy 新版本安装失败，可放宽为：

```bash
python3 -m pip install "catboost>=1.2.5" "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" "scipy>=1.10"
```

## 数据

仓库已包含：

- `data/train.csv`（14930 行，含 `label`）
- `data/test.csv`（6398 行）
- `data/submit_sample.csv`

无需额外下载。

## 验收主交付（必须）

```bash
bash run_super714.sh --verify
```

检查项：

1. `artifacts/super714/best_v1_{oof,test}.npy` SHA-256 与 `manifest.json` 一致  
2. OOF：main≈0.69992 / alt≈0.69770 / fuse≈0.70128  
3. `submissions/submission_super714.csv` 与冻结 `test fuse`（clip 到 \[0.001, 0.999\]）一致  
4. 行数 6398，列 `id,label`

## 单元测试

```bash
python3 -m unittest discover -s tests -v
```

## 从头重训（可选）

仅当你需要重新生成预测，而非验收交付时：

```bash
# 重训冠军双臂（耗时长）
bash run_super714.sh --baseline-only

# 重跑 TE 候选（默认不覆盖主提交，除非过门槛）
bash run_super714.sh
```

门槛定义见 [`TE_GATE_RESULT.md`](TE_GATE_RESULT.md)。当前完整结果：**未通过**，主提交保持 max2。

## 常见问题

**Q: verify 失败说 SHA 不匹配？**  
不要改动 `artifacts/super714/best_v1_*.npy` 与 `submissions/submission_super714.csv`。用 `git checkout --` 还原。

**Q: 自定义数据路径？**  
`DATA_DIR=/path bash run_super714.sh --verify`，目录内需有 `train.csv` 与 `test.csv`。

**Q: real714_pkg 是什么？**  
线上冠军源材料的只读归档；日常复现只跑根目录 `run_super714.sh` 即可。
