# 冲榜计划（相对线上 0.71503 / 榜首 0.72487）

## 当前冠军（立刻可交）

**`submissions/submission_champion.csv`** = **fp_v4**（第一性原理）

| 指标 | 值 |
|---|---:|
| 本地 OOF | **0.71640** |
| nested | **≈0.71660** |
| vs v3 dual | **+0.00924** |
| vs 冻结 AM40 | **+0.0146** |

配方与证据：`docs/FIRST_PRINCIPLES.md`  
复现：`bash run_fp_v4.sh`

```text
score = 0.50 * v3_dual + 0.35 * bits64 + 0.15 * x0_18_q20
```

## 我们发现了什么（不是调参）

1. **忽略的原始列**：`x0..x18` 从未进入 `build_main/alt`  
2. **更丰富的 id**：64 个 bit 平面 TE 池 AUC≈0.591，与 v3 Spearman≈0.03

## 备份

| 文件 | 说明 |
|---|---|
| `submission_am40_idbytes_v3.csv` / 旧 v3 | OOF 0.70717 |
| `submission_am40_idbytes_tempered.csv` | 稠密 V7 |
| `submission_am40_idbytes_safe.csv` | v1 w=0.85 |
| `submission_w62.csv` | 线上锚点 0.71503 |

## 提交纪律

- **主交 fp_v4 champion**  
- Jitter 仅当融合 OOF **> 0.71640** 才替换  
- 已证伪勿再烧：region / Plus / 扩 bags / id 当 CatBoost 类别 / **fold-local 分组统计（见 docs/GROUP_STATS.md）**  
