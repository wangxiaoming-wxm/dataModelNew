# V4-ext（诚实推高阶段最高）

> 提交：`submissions/submission_v4ext.csv`  
> 诚实嵌套（20 block seed）：**0.70358**（sd 0.00019）  
> 对照 V4：0.70303 → **+0.00055**  
> 诚实性：`honesty_passed=true`；目标 0.707 / 0.725：**均未达成**

底座：main V4。并入 opus/zcode 已验证诚实臂（固定树、无早停）：
`merger_ord8`、`v2_cat_alt8`、`gap_v5`。

提交规则（嵌套多数票）：`views_max_v4_mag` =
`max(rank)` over V4 十臂 ∪ `{merger_ord8, v2_cat_alt8, gap_v5}`。

## 用全量实测榜校准的预估

| 提交 | nested | 公开榜 | gap |
|---|---:|---:|---:|
| v4_max3 ★ | 0.70307（含 ES） | **0.71222** | +0.00915 |
| v5_honest | 0.70253 | **0.71207** | +0.00954 |
| v3_max3 | ~0.70226 | 0.71184 | +0.00958 |
| v4_honest | 0.69993 | 0.71104 | +0.01111 |
| main V3 | 0.70124 | 0.71064 | +0.00940 |
| repo V5 | 0.70076 | 0.71035 | +0.00959 |
| V2 | 0.69856 | 0.70878 | +0.01022 |

对本文件 nested **0.70358**，按诚实 gap **0.0092–0.0095** 外推：

**预估公开榜 ≈ 0.7128–0.7131**（期望略高于现冠军 0.71222，不是保证）。

**不能**诚实预估 0.725：Bayes 天花板 ≈0.706 → 即便摸到天花板，LB 上限约 **0.715–0.717**。

## 已拒绝（本阶段实测）

| 方向 | 结果 |
|---|---|
| `mine_noxb_honest` 进 max | full OOF 从 0.70413 → 0.70324；嵌套从不选含它的规则 |
| w12 fast 同种子对照 | OOF 0.69113（待与 main/alt fast 对齐后正式闸门） |

## 复现

```bash
PYTHONPATH=src2:src3:src4 python3 src4/fuse4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --report artifacts/v4_ext/fusion_report_v4ext.json
PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --target 0.707 --out artifacts/v4_ext/audit.json
```
