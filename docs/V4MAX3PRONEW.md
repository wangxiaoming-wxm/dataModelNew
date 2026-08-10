# V4max3proNew

## 一句话结论

忠实移植外部 `715.zip` 语义臂（全量 **5×5×10**，不减配）并融合到 v4max3pro。  
**独立审核 VERDICT：PROTOCOL_RISK**（见 `docs/INDEPENDENT_AUDIT_V4MAX3PRONEW.md`）。

| 项 | 值 |
|---|---:|
| max3 嵌套 | 0.70307 |
| v4max3pro 嵌套 | 0.70522 |
| New 嵌套 | 0.70557 |
| Δ vs max3 | +0.00250（4/5 block 正，较稳） |
| Δ vs pro | **+0.00035**（仅 3/5 正，P≈0.78，不稳） |
| vs pro Spearman | **0.997**（几乎同秩微调） |
| 乐观外推 LB | ≈0.7147（**非已验证公开榜**） |

**外部声称 0.71504：无法核验，不可信。** zip 无提交/融合；本地 semantic nested 仅 **0.69578**。

**提交建议：主推仍用 `submission_v4max3pro.csv`；New 仅探索位，勿包装成 0.715 复现。**

产物文件（探索）：`submissions/submission_v4max3pronew.csv`

## 冻结配方

```text
max(
  rank(merger_ord8),
  rank(v2_cat_alt8),
  rank(ord_noxb_bag),
  rank(plus_strong),
  rank(noxb10),
  rank(semantic_rmse)   # 715 忠实移植
)
→ clip(label, 0.001, 0.999)
```

即：**v4max3pro + semantic_rmse**。

## 715 臂（不减配）

| 项 | 值 |
|---|---|
| FE | `src4/feat_semantic.py`（structured_string + days_condition + dual_category TOP_CROSS + 16 ratios） |
| 模型 | CatBoostRegressor `RMSE`，depth=6，lr=0.03，iter=900，ES=120 |
| 协议 | 5×5×10 bagging，折内 fit |
| pooled OOF | **0.69597**（与 zip 日志量级一致） |
| 产物 | `artifacts/v4max3pronew/semantic_rmse.npz` |

```bash
# 全量训练（耗时长，效果不打折）
python3 src4/train_semantic_rmse.py --seed 2026 --threads 4 --bags 10
# … 2027..2030
python3 src4/train_semantic_rmse.py --merge-only
```

可选多样性臂（同 FE + Logloss，全量 5×5×10 已训完）：

```bash
python3 src4/train_semantic_logloss.py --seed 2026 --threads 4 --bags 10
```

**结论：加入 `semantic_logloss` 后嵌套 0.70521 < 0.70557，不进入正式配方**（产物保留供审计）。

## 复现 / 审核

```bash
python3 src4/build_submission_v4max3pronew.py --check
# 期望 frac_diff=0, ok=True

python3 src4/build_submission_v4max3pronew.py --write
```

报告：
- `artifacts/v4max3pronew/recipe_report.json`
- `artifacts/v4max3pronew/status_report.json`

## 风险

1. ES + 10 bagging 抬高本地 OOF（PROTOCOL_RISK），公开榜不一定同幅度迁移。  
2. 相对 pro 的本地增益较小（+0.00035）；test 相对 pro 仍高度相关（Spearman ≈0.997）。  
3. 乐观 LB 外推假设 max3 的 CV→LB 间隙可迁移，偏乐观。  
4. 715 的 0.71504 无法在本仓复现核验。
