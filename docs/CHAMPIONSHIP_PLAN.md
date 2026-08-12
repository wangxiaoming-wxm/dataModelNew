# 冲榜计划（相对线上 0.71503 / 榜首 0.72487）

## 当前冠军（立刻可交）

**`submissions/submission_champion.csv`** = AM40 + id-bytes TE **v3 dual**

| 指标 | 值 |
|---|---:|
| 本地 OOF | **0.70717** |
| nested (选 w) | **0.70737** |
| vs 冻结 AM40 | **+0.00535** |
| vs W62 本地 OOF | **+0.0056** 量级 |
| vs 榜首缺口（线上） | 仍约 **0.01**（本地增益不能直接平移） |

配方见 `docs/ID_BYTES.md`；复现：`bash run_am40_idbytes.sh`。

## 关于「每个 seed 都要 >0.6999？」

**不需要。** Seed 级 OOF 常见 0.691–0.697；融合吃的是跨 seed 排序稳定性。

## 策略分层

1. **主交（夺冠导向）**：v3 dual idbytes — 当前最强合法正交信号  
2. **备份**：`submission_am40_idbytes_tempered.csv`（纯稠密 V7）/ `*_safe.csv`（v1 w=0.85）  
3. **在跑**：SUPER714-Jitter — 仅当融合 OOF **> 0.70717** 才晋升替换 champion  
4. **已证伪勿再烧提交**：TE 第三臂、region、扩 seed/bags、Plus 改世界、id 当 CatBoost 类别

## 提交纪律

- 有提交次数：交 **champion**  
- 只剩 1 次且极度厌恶风险：tempered → safe  
- 未过门禁的实验：**不要交**，守 W62 / 或交已验证的 champion
