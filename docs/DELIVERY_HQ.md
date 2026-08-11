# 最高质量交付（对齐策略原文 2026-08-11）

> 依据：`复盘_全流程_极详细版_20260811.md` + `下一步策略_20260811.md`  
> 分支：`cursor/beat-max3-6de7`  
> 硬目标：公开榜 **> 0.71222**（若不可达则守住冠军）

---

## 0. 一句话

**不要再交「多臂厨房水槽」**（`max3_best` / `stage_best` / `max3_pro`）。那是 v4ext 失败模式（高相关臂堆叠 → nested 虚高、gap 压缩、LB 回退）。  
**当前唯一策略合规的探索交榜**是 **`submission_max3_plus.csv`**（冠军三臂 + 已验证互补的 `plus_strong`）。  
**守擂默认**仍是 **`submission_v4_max3.csv`（0.71222）**。

---

## 1. 策略硬约束（必须遵守）

| 约束 | 来源 |
|---|---|
| 底座保持 ≤3～4 臂；禁止堆 corr>0.97 同族臂 | 策略 §2 / §5.1 |
| 融合用 `max(rank)` | 复盘 / 冠军协议 |
| 新臂入场：AUC>0.690 **且** vs mo8 corr<0.88 | 策略 §4 方向 A / §6 实验门槛 |
| 禁止 nested+常数 gap 外推 LB | 策略 §5.4；V4ext 实锤 |
| `noxb10` 是 ord 孪生（测试相关 0.9988）→ **禁止再进融合** | 复盘 §3.13 |
| `cat_w12_*` 同族高相关 → **禁止为刷 nested 而堆入** | 策略 §2.3 LOO |

---

## 2. 交榜优先级（监督放行）

| 优先级 | 文件 | 结构 | 本地 nested Δ vs max3 | 策略判定 |
|---:|---|---|---:|---|
| **0 守擂** | `submissions/submission_v4_max3.csv` | mo8+ca8+ord_noxb | 0（锚） | **默认最终成绩** |
| **1 探索** | `submissions/submission_max3_plus.csv` | max3 + **plus_strong** | **+0.00146** | **唯一推荐新交**（plus 测试赢率平衡、真正交臂） |
| 2 备选 | （若探针 admit）`submission_max3_ortho_*.csv` | max3 + 正交探针 | TBD | 仅当 AUC/corr 门禁全过 |
| ❌ 禁止 | `submission_max3_best.csv` / `*_stage_best*` / `*_pro*` | +noxb10/+w12/+多臂 | nested 虚高 | **v4ext 同类风险，勿交** |

说明：此前 `stage_best`（Δ≈+0.003）把新 noxb / w12 / noxb10 堆进 max，**本地好看、迁移危险**；按策略原文应降级为研究产物，不作为交榜。

---

## 3. 正在跑 / 将跑的高质量实验（场景三）

脚本：`src_beat/run_strategy_probes.py`（Ordered / iter2000 / od_wait200，不降配）

| 实验 | 内容 | 入场门槛 |
|---|---|---|
| exp1 | ratio 非线性分段 + source×ratio_bin（折内） | AUC>0.690 且 corr(mo8)<0.88 |
| exp2 | condition 悬崖 × source | 同上 |
| exp3 | 低 ratio 子群残差纠正器 | 纠正后 AUC>0.690 且 corr<0.88 且残差 AUC>0.55 |

**不过门槛 → 不进 max，确认 0.716 不可达路径关闭。**

库存建议（策略 §8.2）：若本地能拿到 `v6_zcode`，其为 3 臂同协议探索（预期 LB~0.712 噪声带）；本环境 opus 包未含 v6 产物，故以探针替代。

---

## 4. 训练调度纠正

| 原计划（已停） | 原因 |
|---|---|
| HQ 堆 depth8/slow7/b1/w12/goldmine 进同一 max | 高相关堆臂 = v4ext |
| `refresh_stage_best` 自动交多臂最优 | 优化错目标（nested 而非可迁移结构） |

| 现计划 | |
|---|---|
| P1 `ord_noxb_new16` | 允许跑完；**只作同槽去噪研究，默认不作为第 4 臂提交** |
| 策略探针 exp1–3 | P1 结束后全力跑 |
| 监督 | 仅放行 ≤4 臂且 corr 门禁通过的配方 |

---

## 5. 诚实结论（写进交付）

- **0.716**：策略论证在现有数据+CatBoost 下**极可能不可达**（先验>90%）。
- **超过 0.71222**：只能赌「真正正交臂」；历史 5 轮未出现；探针是最后合规尝试。
- **不虚报**：任何 nested 增益若伴随臂数↑且 avg_corr↑，默认视为**选择偏差**，不宣称 LB 必升。

---

## 6. 你怎么交

1. **若要稳**：交 `submission_v4_max3.csv`（已是冠军）。  
2. **若要探索一次**：交 `submission_max3_plus.csv`。  
3. **等探针**：看 `artifacts/beat_max3/probes/summary.json` 的 `admit_to_max`；仅 `true` 时交对应 `submission_max3_ortho_*.csv`。

## 7. 当前可交最高版（自动刷新）

| 文件 | 配方 | nested Δ |
|---|---|---:|
| **`submissions/submission_beat_max3.csv`** | max(mo8, ca8, **ord_strong**, plus) | **见 report** |
| `submissions/submission_ship_max3s_plus.csv` | 同上显式名 | |
| `submissions/submission_ship_max3_plus.csv` | 原 max3+plus | +0.00146 |

`ord_strong` = 0.5·冠军 `ord_noxb_bag` + 0.5·新种子袋（**同一逻辑臂去噪，不是堆孪生臂**）。
