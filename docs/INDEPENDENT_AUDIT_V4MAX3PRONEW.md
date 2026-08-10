# 独立审核：V4max3proNew

**审核角色**：与实现 agent 隔离的独立复算。  
**日期**：2026-08-10  
**VERDICT**：**PROTOCOL_RISK**

## 一句话

本地 nested 数字可复算、提交 bit 一致；相对 pro 的诚实增益极小且块级不稳；**0.71504 无法核验，不可当作真**。主提交建议仍用 **v4max3pro**（或更保守 max3），不要把 New 包装成 0.715 复现。

## 真正诚实数字（从 OOF/CSV 复算）

| 对象 | raw OOF | nested 5-block | Δ vs max3 | Δ vs pro | vs max3 Spearman | vs pro Spearman |
|---|---:|---:|---:|---:|---:|---:|
| max3 | 0.70336 | **0.70307** | — | — | ~1 | 0.99174 |
| pro | 0.70572 | **0.70522** | +0.00215 | — | 0.99174 | — |
| New | 0.70607 | **0.70557** | +0.00250 | **+0.00035** | 0.98920 | **0.99727** |
| semantic_rmse | 0.69597 | **0.69578** | −0.00729 | −0.00944 | 0.96387 | 0.96861 |

- New vs max3 逐块 Δ：4/5 正；块 bootstrap P(Δ>0)≈0.99  
- New vs pro 逐块 Δ：仅 3/5 正；P(Δ>0)≈**0.78**（CI 含 0）  
- 标签置换 sanity：≈0.5（未见标签泄漏）  
- `python3 src4/build_submission_v4max3pronew.py --check` → `frac_diff=0`

复算入口：`python3 src4/audit_v4max3pronew_independent.py`  
摘要：`artifacts/v4max3pronew/independent_audit_summary.json`

## 过拟合 / 协议风险（按严重度）

1. **高**：semantic = ES + 5×5×10 bagging；plus/noxb10 = 10-fold ES → 本地 OOF 系统性偏乐观。  
2. **高**：相对 pro 仅 +0.00035，test Spearman 0.997，实质是 pro 微调；公开榜收益接近噪声。  
3. **中高**：配方枚举多候选后“刚过 pro”才 ADMIT → 挑选乐观。  
4. **中**：semantic 单臂 nested 0.69578 ≪ 融合 0.70557；`max3+sem` 几乎无增益。  
5. **中**：noxb10 ↔ ord_noxb_bag 高度共线（OOF≈0.992 / TE≈0.999）。  
6. **中**：semantic 与既有臂 Spearman 0.93–0.97，非强正交。

## 关于 0.71504

**无法核验 / 若当作已验证则不可信。**  
715.zip 仅有单臂代码（本地 ~0.696），无提交 CSV、无融合、全文无 0.71504。本仓乐观外推 ≈0.7147 也**不是**已验证公开榜。

## 提交建议

| 优先级 | 文件 | 理由 |
|---|---|---|
| **主推** | `submission_v4max3pro.csv` | 相对 max3 有更清晰本地证据；New 增益不稳 |
| 保守 | `submission_v4_max3.csv` | 公开榜已锚 0.71222；协议更干净 |
| 探索位 | `submission_v4max3pronew.csv` | 仅作高相关微调探索，**勿声称 0.71504** |
