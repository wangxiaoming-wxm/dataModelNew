# task_w62 — 加权融合提交（可复核）

本分支交付 **W62**：在冻结 best_v1 双臂上做 `0.62*main + 0.38*alt` 融合。

| 项 | 值 |
|---|---|
| 提交文件 | [`submissions/submission_w62.csv`](submissions/submission_w62.csv) |
| 本地 OOF | **0.70159** |
| 线上 AUC | **0.71503**（已提交确认） |
| 对照 max2 | 0.70128（`submission_super714.csv`） |
| 耗时 | **秒级**（不重训） |

## 快速复核

```bash
git checkout task_w62
pip install -r requirements.txt
bash run_w62.sh --verify
```

完整说明：[`docs/W62.md`](docs/W62.md)

## 说明

- 与公开冠军 **max2 提交文件不同**（融合规则不同），便于比赛另交。  
- 不包含 / 不影响 `submission_super714_plus` 的长训练；Plus 在其他分支进行。  
