# vz20：诚实尝试超越 vz19（结论：可用正交信号已被 vz19 榨干）

保险理赔二分类 AUC。目标：构建 vz20 相对 vz19 在诚实 held-out 下提升 ≥ +0.0015。

## 结果（TL;DR）
| 指标 | 值 |
|---|---|
| 协议 | held-out 5 折，全新 outer seed **90210**，4 seeds×800 trees，预注册权重 |
| vz19 fold-mean | 0.69957 |
| vz20 fold-mean | 0.69978 |
| **lift vs vz19** | **+0.00021**（门禁 +0.0015）→ ❌ 未过 |
| 不退化外折 | 2/5（门禁 ≥3/5）→ ❌ |
| 置乱哨兵 | 0.5128/0.5129 ∈[0.48,0.52] → ✅ 无泄漏 |
| oracle 融合上界 | +0.0002（作弊调权重也到不了 +0.0015） |
| **推荐提交** | **仍为 vz19**；vz20 与其统计不可区分，不建议替换 |

详细裁决见 `STATUS.md`，证据见 `docs/EVIDENCE.md`，归因见 `docs/METHOD.md`，预注册见 `docs/PROTOCOL.md`。

## 为什么没超越（第一性原理）
- rich/freq 特征与 vz19 的手工交叉**近重复**：ratio_rich 对 vz19 arm1 的 Spearman≈0.985。
- vz19 的 **max2 在干净 seed 上增益≈0**（−0.00007），其文档 +0.00176 是 OOF 自采样乐观。
- **byte07 是唯一真实正交弱信号(+0.00075)**，已被 vz19 吃满；协议禁止扩展更多 id 字节。
- 因此诚实融合天花板≈+0.0002，无法达到 +0.0015。

## 复现
```bash
# 1) 训练 8 臂 + byte07，缓存 per-fold held-out 预测（~75 min, 4 核）
python3 src/vz20_arms.py --profile full --outer-splits 5 --nseed 4 --trees 800 \
    --arms A1,A2,REF1,REF2,R1,R2,R3,R4

# 2) 用预注册配方装配并评估 vz19 vs vz20（读缓存，秒级）
python3 src/vz20_combine.py --profile full --outer-splits 5

# 3) 置乱哨兵
python3 src/vz20_permutation.py --arm R1 --folds 3 --nseed 2 --trees 600

# 4) 锁定配方 -> 全 train 重拟合 -> submission_vz20.csv
python3 src/vz20_final.py --nseed 4 --trees 800
```

## 文件
```
vz20/
├── README.md / STATUS.md
├── docs/{PROTOCOL,METHOD,EVIDENCE}.md
├── src/{features.py, vz20_arms.py, vz20_combine.py, vz20_permutation.py, vz20_final.py, ...}
├── ref_rebuild/{features.py, models.py, evaluation.py}   # rebuild V2 的 rich/freq 特征来源
├── artifacts/vz20/{metrics.json, permutation.json, final_sha.json, cache/*.npy}
├── submission_vz19.csv     # 参考基线提交
└── submission_vz20.csv     # 本轮诚实交付（与 vz19 近等价，不建议替换）
```

## 诚实声明
本轮**未过 +0.0015 门禁**。按任务要求不注水：如实报告 vz20≈vz19，并给出可复现的无偏 harness 与
证据。若只能交一次，仍交 **vz19**。
