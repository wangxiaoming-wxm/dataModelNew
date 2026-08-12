# 冲榜计划（相对线上 0.71503 / 榜首 0.72487）

## 当前冠军（立刻可交）

**`submissions/submission_champion.csv`** = **fp_v5 and_heavy**

| 指标 | 值 |
|---|---:|
| 本地 OOF | **0.72880** |
| nested fold-mean | **0.72900** |
| vs fp_v4 | **+0.0124**（nested） |
| sha256 | `2e01cad679b8b13f21fbe6f6e781e2eb265526fbbd955789bf240a3fa979c889` |

```text
0.30*v3_dual + 0.25*bits64 + 0.10*x0_18 + 0.35*and_all
```

`and_all` = 选择无关的 id bit-AND 二阶 TE 池（within-byte + cross-byte same-bit）。

复现：`bash run_fp_v5.sh` / `--verify`  
证据：`docs/FIRST_PRINCIPLES.md`、`artifacts/first_principles/v5_metrics.json`

## 备份

| 文件 | 说明 |
|---|---|
| `submission_fp_v5_tempered.csv` | and 权重更低（更贴 v3） |
| `submission_fp_v5_aggressive.csv` | and_max，本地更高、更偏 id |
| `submission_fp_v4.csv` | 上一版 |
| `submission_w62.csv` | 线上锚点 0.71503 |

## 已证伪（勿再烧提交）

region / Plus / 扩 bags / id 当 CatBoost 类别 / group-stats / Jitter main（池 OOF 0.698 < 冻结 main） / **Lossguide**（3fold +0.0039 为噪声，见 `docs/LOSSGUIDE.md`） / **特征剪枝**（solo +0.0017 但融不进 fp_v5，见 `docs/FEATURE_PRUNE.md`）

## 提交纪律

- **主交 fp_v5 champion**
- 极保守：tempered；搏高：aggressive（风险更大）
