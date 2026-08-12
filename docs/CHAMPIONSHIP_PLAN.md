# 冲榜计划（相对线上 0.71503 / 榜首 0.72487）

## 当前冠军（立刻可交）

**`submissions/submission_champion.csv`** = **fp_v7 cross30**

| 指标 | 值 |
|---|---:|
| 本地 OOF | **0.75838** |
| nested fold-mean | **0.75851** |
| vs fp_v6 heavy_xor | **+0.0150**（nested） |
| sha256 | `02e83afbc86b07c2beaa80922370387fa3d0ed652116ed12d77ead1fe7a3f1d7` |

```text
cmean = 0.5*crossOR + 0.5*crossXOR
0.15*v3 + 0.10*bits + 0.05*xs + 0.22*and_all
+ 0.06*tri + 0.06*or + 0.06*xor + 0.30*cmean
```

复现：`bash run_fp_v7.sh` / `--verify`  
证据：`docs/FIRST_PRINCIPLES.md`、`artifacts/first_principles/v7_metrics.json`

## 备份

| 文件 | 说明 |
|---|---|
| `submission_fp_v7_tempered.csv` | = fp_v6 heavy_xor（更贴表格臂） |
| `submission_fp_v7_aggressive.csv` | 0.5*heavy + 0.5*cmean，本地更高、更偏 id |
| `submission_fp_v6.csv` | 与 tempered 同内容 |
| `submission_fp_v5.csv` / `submission_fp_v4.csv` | 历史档 |
| `submission_w62.csv` | 线上锚点 0.71503 |

## 已证伪（勿再烧提交）

region / Plus / 扩 bags / id 当 CatBoost 类别 / group-stats / Jitter / **Lossguide**（见 `docs/LOSSGUIDE.md`） / **特征剪枝**（见 `docs/FEATURE_PRUNE.md`）

## 提交纪律

- **主交 fp_v7 champion**（本地大幅抬升，但更偏 id TE；线上方差可能更大）
- 更保守：tempered（=v6）；搏高：aggressive
- 本地 nested OOF ≠ 线上排名；勿交公开克隆 `submission_super714.csv`
