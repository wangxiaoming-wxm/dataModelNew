# id 字节信号 → AM40+idbytes v3 dual（当前冠军）

## 能不能用？

**能。** `id` 是原始字段；解析 hex 字节/半字节做 fold-local TE，合法。

## 关键结论

| 做法 | 结果 |
|---|---|
| id 字节当 CatBoost 类别特征 | **不涨**（甚至变差） |
| fold-local TE 再与 AM40 混合 | **大涨**（与 AM40 近乎正交） |

## v3 dual 融合（当前交付 / 夺冠主交）

```text
TE_SEEDS = (2026, 7, 42, 99, 314, 2718)
V7 = [b3, b2hi, b1hi, b0, b4hi, b5, b7, b7hi, b5hi, b6hi]   # 稠密
V2 = [b0, b4, b5, b7, b2hi, p47]                           # 旧 v2
id_pool = 0.70 * mean_rank(TE_multiseed(V7)) + 0.30 * mean_rank(TE_multiseed(V2))
score   = 0.55 * AM40 + 0.45 * id_pool
```

| 方案 | OOF | nested (w) |
|---|---:|---:|
| W62 | 0.70159 | — |
| AM40 | 0.70181 | — |
| idbytes v1 (4 bytes, w=0.85) | ~0.7039 | — |
| idbytes v2 (单 seed) | 0.70648 | 0.70639 |
| idbytes v7 dense (w=0.55) | 0.70667 | 0.70686 |
| **idbytes v3 dual（主交）** | **0.70717** | **0.70737** |

相对冻结 AM40：本地 **+0.00535**；相对 W62 OOF：**+0.0056** 量级。

```bash
bash run_am40_idbytes.sh
bash run_am40_idbytes.sh --verify
```

主文件：`submissions/submission_champion.csv`（=`submission_am40_idbytes.csv`）

## 备份档位

| 文件 | 配方 | 用途 |
|---|---|---|
| `submission_am40_idbytes_tempered.csv` | 纯 V7 dense, w=0.55 | 去掉稀疏 p47 |
| `submission_am40_idbytes_safe.csv` | V1 bytes, w=0.85 | 极保守 |

## 提交建议

次数紧时优先交 **v3 dual champion**（当前最高 nested / win-prob）。  
Jitter 全量若融合 OOF **明确超过 0.70717** 再考虑替换；否则不要用旧同构调参浪费提交。
