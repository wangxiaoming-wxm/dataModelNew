# 策略还原（原件不在本环境）

用户指定路径：
- `/Volumes/pssd/app/ml/正式比赛/20260810-cursor-opus5/复盘_全流程_极详细版_20260811.md`
- `/Volumes/pssd/app/ml/正式比赛/20260810-cursor-opus5/下一步策略_20260811.md`

**本 Cloud 环境、`/tmp/audit_all/opus`、zip、git 均未找到上述原件。**  
以下根据 opus `v4_max3/README`、公开榜回执、V4ext 失败与本地门禁结果还原可执行策略。

## 硬目标
公开榜 **> 0.71222**（`submission_v4_max3.csv`）。

## 已证实失败
- V4ext（丢 `ord_noxb_bag` + 诚实 nested 外推）→ **0.71123**

## 必守协议
1. 底座三臂冻结：`merger_ord8` + `v2_cat_alt8` + `ord_noxb_bag`
2. 融合：`max(rank)`（mean/stack 已验证更差）
3. 允许混合 ES（与冠军同协议）；nested 乐观，**禁止 +0.0095 外推 LB**
4. 新臂须改 test 序：Spearman∈[0.985,0.997]、Δnested≥0.001、blocks+≥4/5

## 高质量训练方针（用户强制：不因耗时降配）
| 阶段 | 内容 | 规格 |
|---|---|---|
| P1 | 新种子 noxb（冠军配方） | depth7 Ordered ES iter1200 × **16 seeds** |
| HQ-P2 | 更深/更慢变体 | depth8×16seeds iter**2000**；slow7 iter**2500** lr0.015 |
| HQ-P3 | b1 视图 | 16 seeds iter**2000** |
| HQ-P4 | plus 族扩训 | H2/H3 级 iter2500 ×8 |
| HQ-P5 | goldmine/CoFEH 叠 B5 | 各 8 seeds Ordered |

## 当前阶段最优
见 `docs/BEAT_MAX3.md` / `submission_max3_stage_best.csv`。

## 入口
```bash
bash run_beat_max3_watch_hq.sh   # P1 结束后自动切高质量续训
```
