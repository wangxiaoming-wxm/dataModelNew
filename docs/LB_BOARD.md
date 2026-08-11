# 公开榜实测板（用户回执）

| 时间 | 公开榜 | 文件 | 备注 |
|---|---:|---|---|
| 08-11 | **0.71123** | submission_v4ext.csv | nested 0.70381；预估 0.713 **失败**（高估≈0.002） |
| 08-10 21:41 | 0.71207 | submission_v5_honest.csv | |
| 08-10 07:41 | **0.71222 ★** | submission_v4_max3.csv | **仍为冠军** |
| 08-10 07:39 | 0.71035 | submission_v5.csv | |
| 08-09 22:35 | 0.71064 | submission_v3.csv | |
| 08-09 16:20 | 0.71184 | submission_v3_max3.csv | |
| 08-09 12:59 | 0.70878 | submission_v2.csv | |
| 更早 | ≤0.70722 | B7 / v10 等 | |

**教训**：诚实 nested 外推不可靠。冲超 max3 必须以 max3 三臂为底座，融合必须是 `max(rank)`。

## 冲超候选（本地门禁已过，待交榜）

| 文件 | nested Δ vs max3 | Spearman | 备注 |
|---|---:|---:|---|
| **submission_max3_best.csv** | **+0.00282** | 0.9918 | plus+noxb10+cat_w12_d5；**首选交榜** |
| submission_max3_pro.csv | +0.00215 | 0.9917 | plus+noxb10 |
| submission_max3_plus_w12.csv | +0.00238 | 0.9917 | plus+w12 |
| submission_max3_pro_sem.csv | +0.00250 | 0.9892 | plus+noxb10+semantic_rmse |
| submission_max3_plus.csv | +0.00146 | 0.9917 | 最小改动 |

详见 `docs/BEAT_MAX3.md`。未回执前不宣称已超过 0.71222。
