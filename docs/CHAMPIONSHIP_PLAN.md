# 冲榜计划（相对线上 0.71503 / 榜首 0.72487）

## 当前冠军（立刻可交）

**`submissions/submission_champion.csv`** = **fp_v8 byte20**

| 指标 | 值 |
|---|---:|
| 本地 OOF | **0.77016** |
| nested fold-mean | **0.77033** |
| vs fp_v7 | **+0.0118**（nested） |
| sha256 | `fa2a06f4ed9af099e37225a2a9ff2e10e0a0a7979ac9c7e5dfacdcdde5866d67` |

```text
bytepair_mean = 0.5*pool(byteXOR) + 0.5*pool(byteAND)
score = 0.80*v7_cross30 + 0.20*bytepair_mean
```

复现：`bash run_fp_v8.sh` / `--verify`  
证据：`docs/FIRST_PRINCIPLES.md`、`artifacts/first_principles/v8_metrics.json`

## 备份

| 文件 | 说明 |
|---|---|
| `submission_fp_v8_tempered.csv` | = fp_v7 cross30 |
| `submission_fp_v8_aggressive.csv` | 0.70*v7+0.30*bytepair |
| `submission_fp_v7.csv` / `submission_fp_v6.csv` | 历史档 |
| `submission_w62.csv` | 线上锚点 0.71503 |

## 已证伪（勿再烧提交）

region / Plus / 扩 bags / id 当 CatBoost 类别 / group-stats / Jitter / Lossguide / 特征剪枝 / logistic 叠臂 / 等权多臂 rank-mean（乐观偏置）

## 提交纪律

- **主交 fp_v8 champion**（本地继续抬升，更偏 id TE）
- 更保守：tempered（=v7）；再保守：v6/v5
- 本地 nested OOF ≠ 线上排名；勿交公开克隆 `submission_super714.csv`
