> 交付入口：[`docs/DELIVERY.md`](DELIVERY.md) · 提交文件 `submissions/submission_v4ext.csv`

# 诚实推高状态（对照你的全量公开榜）

更新：2026-08-10（分支 `cursor/honest-push-v4ext-6de7`）

## 你的实测榜（硬锚）

| 文件 | 公开榜 |
|---|---:|
| submission_v4_max3.csv | **0.71222 ★** |
| submission_v5_honest.csv | 0.71207 |
| submission_v3_max3.csv | 0.71184 |
| submission_v4_honest.csv | 0.71104 |
| submission_v3.csv | 0.71064 |
| submission_v5.csv | 0.71035 |
| submission_v2.csv | 0.70878 |
| … | ≤0.70722 |

## 本轮交付（建议下一个未测提交）

| 项 | 值 |
|---|---|
| 文件 | `submissions/submission_v4ext.csv` |
| 诚实 nested | **0.70381–0.70382** |
| 相对 V4 | +0.00079 |
| honesty_passed | true |
| 预估 LB | **0.7130–0.7133**（gap 0.0092–0.0095） |
| 相对现冠军 | 期望约 +0.0008～+0.0011（**非保证**） |

配方：V4 十臂 ∪ opus `{merger_ord8,v2_cat_alt8}` ∪ `cat_w12_d5`(8seed) 的 `max(rank)`。

## 对 0.725 的裁决（不骗你）

| 量 | 值 |
|---|---:|
| 要 LB 0.725 且 gap≈0.0095 | 需 nested ≈ **0.7155** |
| 当前诚实 nested | **0.7038** |
| 当前 Bayes（isotonic） | ≈ **0.7075** |
| 诚实 LB 上限（Bayes+gap） | ≈ **0.715–0.717** |
| 距 0.725 | ≈ **0.008–0.010** |

**不能诚实承诺 0.725。**  
缺口大于「再堆编码多样性」的历史幅度（V3→V4 整段只 +0.0018；本轮全部努力合计 +0.0008）。  
继续工作只会在 0.704～0.706 附近抠噪声，除非出现**能抬高 Bayes 的新标签相关信号**。

## 本轮已诚实试过并拒绝

| 方向 | 结果 |
|---|---|
| mine_noxb_honest | 拖 max |
| w12 d6l6 | nested 下降 |
| Ordered×alt/w12 | 弱于 Plain |
| PairLogit | OOF≈0.65 |
| RMSE 固定树 | bag≈0.690 |
| 语义 FE 固定树（715 配方去 ES） | OOF≈0.682 |
| 8-seed vs 4-seed w12 | 嵌套持平 |

## 仍在跑 / 刚拒

- w12 × 20-fold：bag 0.70003，**进规则后 nested 下降 → 拒**
- 已知杠杆（折多样 / Ordered / RMSE / PairLogit / 语义FE / 弱 noxb）已穷尽拒完
- 下一方向：只做「可能抬 Bayes」的新 label-free 信号，不再堆同质种子

## 建议你现在做的事

1. **提交 `submission_v4ext.csv`** 拿真实榜，校准本轮预估。  
2. 若榜 <0.7120：先核对文件字节，再决定是否停。  
3. 若榜在 0.7125–0.714：与预估一致，说明本地尺子仍准，但 **0.725 仍不可达**。  
4. 不要交 pro/pronew/ES 配方来「冲 0.725」——那是换尺子，不是诚实达到。

## 规划终裁（Bayes 抬升）

独立规划结论：**诚实协议下无可行 Bayes 抬升路径**可达 0.725。

要点：需 Bayes ≥~0.715，当前 ≈0.7075；信号空间近邻同标 lift 仅 +0.066；已知改风险函数杠杆已实测拒完；历史增益幅度比缺口小一个数量级。

**唯一诚实下一步**：提交 `submission_v4ext.csv` 校准 gap；若落在 0.7125–0.714 预估带，停止冲 0.725。
