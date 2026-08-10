# V4max3pro

## 一句话结论

相对冻结基线 `submission_v4_max3.csv`（公开榜 **0.71222**）做了增量融合，本地嵌套 OOF **0.70522**（+0.00215）。  
按 max3 的 CV→LB 间隙外推约 **0.7144**。**不能诚实声称会到 0.7155。**  
独立审核结论为 **PROTOCOL_RISK**（无作弊/无标签泄漏，但含 ES/10 折乐观与选择风险）。

## 冻结配方

```text
max(
  rank(merger_ord8),   # artifacts/v4max3/merger_ord8.npz   honest
  rank(v2_cat_alt8),   # artifacts/v4max3/v2_cat_alt8.npz   honest
  rank(ord_noxb_bag),  # artifacts/v4max3/ord_noxb_bag.npz  ES
  rank(plus_strong),   # artifacts/v4max3pro/plus_strong.npz  10-fold ES family
  rank(noxb10)         # artifacts/v4max3pro/noxb10.npz       10-fold ES, 8 seeds
)
→ clip(label, 0.001, 0.999)
```

提交文件：**仅此一个**正式候选  
`submissions/submission_v4max3pro.csv`

基线对照：  
`submissions/submission_v4_max3.csv`

## 如何复现 / 审核（下载本分支后）

只需已提交的臂产物 + 构建脚本，**不必重新长训**：

```bash
python3 src4/build_submission_v4max3pro.py --check
# 期望：frac_diff=0, max_abs ~ 1e-16 量级, ok=True

python3 src4/build_submission_v4max3pro.py --write
# 重写 submission_v4max3pro.csv 与 artifacts/v4max3pro/recipe_report.json
# 再 --check 必须仍通过
```

报告：
- `artifacts/v4max3pro/recipe_report.json`（构建脚本生成，与 CSV 同源）
- `artifacts/v4max3pro/status_report.json`（摘要）

## 数字（与脚本一致）

| 项 | 值 |
|---|---:|
| max3 嵌套 OOF | 0.70307 |
| max3 公开榜 | 0.71222 |
| CV→LB 间隙 | +0.00915 |
| V4max3pro 嵌套 OOF | 0.70522 |
| Δ vs max3 | +0.00215 |
| 乐观外推 LB | ≈0.71437 |
| vs max3 提交 Spearman | ≈0.9917 |

## 臂说明（诚实标签）

| 臂 | 协议标签 | 备注 |
|---|---|---|
| merger_ord8 | honest | 固定树、5 折、Ordered + v2 主特征 |
| v2_cat_alt8 | honest | 固定树、5 折、alt 编码世界 |
| ord_noxb_bag | es | B5 no-xbin Ordered，早停；OOF 可乐观，test 未看标签 |
| plus_strong | plus10 | V10 plus 家族（10 折 ES）强化袋 |
| noxb10 | plus10 | 同族 10 折 ES，8 seed；与 ord_noxb_bag 高度共线 |

## 明确不是最终配方的内容

下列训练脚本曾用于探索，**不进入最终提交**，保留仅供对照：

- `src4/train_main10.py` / `train_hybrid10.py` / `train_alt10.py` / `train_plus5.py` / `train_es_arm.py` / `train_alt2fix.py`
- 旧扫描器 `src4/fuse_v4max3pro.py`（已被 `build_submission_v4max3pro.py` 取代为正式入口）

若需从种子重训 `noxb10` / plus 族，见 `src4/train_noxb10.py`、`src4/train_plus10.py`、`src4/merge_parts.py`（耗时长，审核复现 CSV **不需要**这一步）。

## 风险（提交前必读）

1. ES / 10 折臂会抬高本地 OOF；公开榜不一定同幅度迁移。  
2. `plus_*` 来自 B7 家族，历史 CV→LB 间隙小于 max3；外推 0.714x 偏乐观。  
3. `noxb10` 与 max3 第三臂高度相关，本地增益里有一部分几乎不改变 test 排序。  
4. 今天若只剩 1 次提交且目标必须是 **0.7155**：**不建议交这版**。
