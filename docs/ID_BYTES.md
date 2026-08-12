# id 字节信号 → AM40+idbytes（当前冠军）

## 能不能用？

**能。** `id` 是原始字段；解析 hex 字节做 fold-local TE，合法。

## 关键结论

| 做法 | 结果 |
|---|---|
| id 字节当 CatBoost 类别特征 | **不涨**（甚至变差） |
| fold-local TE 再与 AM40 混合 | **大涨**（与 AM40 近乎正交） |

## v2 融合（当前交付）

```text
specs = [b0, b4, b5, b7, b2hi, p47]
id_pool = mean_rank(TE_5fold(spec))   # auc<0.5 则翻转
score   = 0.65 * AM40 + 0.35 * id_pool
```

| 方案 | OOF | nested OOF |
|---|---:|---:|
| W62 | 0.70159 | — |
| AM40 | 0.70181 | — |
| idbytes v1 (4 bytes, w=0.75) | 0.70441 | 0.70408 |
| **idbytes v2** | **0.70648** | **0.70639** |

```bash
bash run_am40_idbytes.sh
bash run_am40_idbytes.sh --verify
```

主文件：`submissions/submission_am40_idbytes.csv`（= `submission_champion.csv`）

## 提交建议

次数紧时优先交 **idbytes v2**（相对 W62 本地 +0.0049）。线上仍有落差风险，但这是目前最强合法增量。
