# 下一份该交什么（vz19 / W62 已交过，禁止再交）

已交过且打到天花板附近：

| 已交 | 线上 |
|---|---:|
| W62 | 0.71503 |
| vz19 | 0.71298 |

再交 AM40 / rebuild V2 / 本仓 vz20 都和上面 Spearman ≥ 0.987，**换汤不换药，还会停在倒数。**

## 现在交这一份

**文件：** `vz20/next_submit/submission_fp_v8_champion.csv`  
**SHA256：** `fa2a06f4ed9af099e37225a2a9ff2e10e0a0a7979ac9c7e5dfacdcdde5866d67`  
**来源：** `cursor/super714-plus-edb2` 的 `submissions/submission_champion.csv`（= `submission_fp_v8.csv`）  
**和 vz19 的 Spearman：** **0.917**（仓里唯一明显不同的家族）  
**本地声称：** OOF 0.77016 / 所谓 nested 0.77033（id 字节对 XOR/AND 重 TE）

这是仓库里**唯一**本地分数落在冠军 0.749 附近、且你几乎肯定还没交过的提交。

## 风险（必须看）

这条是 **id 重 TE**，不是 CatBoost 业务特征。所谓 nested 很可能偏乐观；线上可能冲到前三，也可能掉下去。  
你已经在诚实 CatBoost 族交到倒数第二，再交同族没有意义。这一发是换轴赌博，不是稳分。

## 备选（仍不要交 vz19/W62）

| 文件 | SHA256 | vs vz19 | 何时用 |
|---|---|---:|---|
| `submission_fp_v8_tempered.csv`（= fp_v7） | `02e83afbc86b…` | 0.960 | 想略保守，仍是 id 轴 |
| `submission_fp_v8_aggressive.csv` | `8eeb0db05497…` | **0.848** | 最不像 vz19，风险最大 |
| `submission_am40.csv` | `de0d337f0287…` | 0.996 | **不要交**，≈ W62 |

一发机会：交 **fp_v8 champion**。
