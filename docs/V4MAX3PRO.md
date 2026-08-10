# V4max3pro 交付说明（实事求是）

## 结论（先看这里）

**不能诚实声称会到公开榜 0.7155。**

| 项 | 数值 |
|---|---:|
| 基线 `submission_v4_max3.csv` 公开榜 | **0.71222** |
| 基线本地嵌套 OOF | **0.70307** |
| 同口径 CV→LB 间隙 | **+0.00915** |
| 要冲 0.7155 约需嵌套 | **≈0.70635**（还需 +0.00328） |
| 本轮最佳嵌套 | **0.70513**（+0.00206） |
| 若间隙不变的预期 LB | **≈0.71428** |
| 是否达到 0.7155 证据门槛 | **否** |

今天只剩 1 次提交时的建议：

1. **若目标必须是 0.7155**：建议**不要交**这一版（证据不够，交了也是赌噪声）。
2. **若接受“相对 max3 小幅抬期望、但不保证”**：可交 `submissions/submission_v4max3pro.csv`（配方见下），预期大约 **0.7135–0.7145**，且存在 plus 臂可能压缩 CV→LB 间隙的风险（B7 历史间隙只有 +0.0045）。

## 配方

冻结 max3 三臂，再加两只新训练/强化臂：

```text
max(
  merger_ord8,          # 冻结，诚实 Ordered 主世界
  v2_cat_alt8,          # 冻结，诚实 alt 世界
  ord_noxb_bag,         # 冻结，ES B5-noxb（max3 原第三臂）
  plus_strong,          # V10 plus 4seed + 新 Plain10 3seed（剔除坏种子 2033）
  noxb10                # 新训：B5-noxb Ordered 10 折 ES，4 seed
)
```

文件：`submissions/submission_v4max3pro.csv`  
报告：`artifacts/v4max3pro/status_report.json`

## 试过但无效 / 负向（避免再走）

| 尝试 | 结果 |
|---|---|
| V5 编码世界加入 max3 | 嵌套下降（与 V5 榜上 0.71035 一致） |
| 把 plus 改成 5 折重训 | 单臂变弱，拖累 max3 |
| merger_ord 的 5 折 ES 升级 | 弱于诚实 fixed-tree |
| hybrid（主特征+x0–x18） | 与 max3 过共线，几乎无增益 |
| alt10 | 几乎无增益 |
| 继续堆 m10/ob/plus_ord | 超过 `max3+plus_strong+noxb10` 后开始掉 |

## 风险（必须说清）

- 本地嵌套里的 `ord_noxb_bag` / `noxb10` / `plus_*` 含 **ES 或 10 折**，OOF 可能偏乐观；测试预测仍未看测试标签。
- plus 来自 B7 家族；B7 当年本地→榜间隙明显小于 max3。**把 plus 加进 max3 可能让真实公开榜低于“嵌套+0.00915”的乐观外推。**
- 因此 **0.71428 是乐观点估计，不是承诺。**

## 复现

```bash
# 冻结臂已在 artifacts/v4max3/
# 新臂：
python3 src4/train_noxb10.py --seed SEED --folds 10
python3 src4/train_plus10.py --seed SEED --boosting Plain --folds 10
python3 src4/merge_parts.py --pattern 'part_noxb10_s*.npz' --out noxb10 --pool rank
python3 src4/build_plus8.py   # 或使用已写入的 plus_strong.npz
python3 src4/fuse_v4max3pro.py --write
```
