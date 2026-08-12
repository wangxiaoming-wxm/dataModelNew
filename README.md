# task-am40 — AM40 混合融合提交（可复核）

本分支交付 **AM40**：在冻结 best_v1 双臂上做

```text
0.40 * max(main, alt) + 0.60 * (0.62*main + 0.38*alt)
```

| 项 | 值 |
|---|---|
| 提交文件 | [`submissions/submission_am40.csv`](submissions/submission_am40.csv) |
| 本地 OOF | **0.70181135** |
| 对照 W62 | 0.70159366（Δ ≈ +0.000218；W62 线上 0.71503） |
| 对照 max2 | 0.701275（`submission_super714.csv`） |
| 耗时 | **秒级**（不重训） |

## 快速复核

```bash
git checkout task-am40
pip install -r requirements.txt
bash run_am40.sh --verify
```

完整说明：[`docs/AM40.md`](docs/AM40.md)

## 说明

- 与 W62 / max2 **提交文件不同**（融合规则不同）。  
- 本分支只含融合验收；不包含 Plus / Bags 等长训练（那些在其他分支进行）。  
