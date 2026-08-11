# V4-ext 说明

> **提交：`submissions/submission_v4ext.csv`**  
> 诚实嵌套（20 block seed）：**0.70381**  
> 轨迹：V4 0.70303 → V4∪opus 0.70358 → **+w12 0.70381**  
> `honesty_passed=true`；**不能诚实承诺公开榜 0.725**

## 配方

嵌套多数票规则：`views_max_v4_ma_w12` =

```text
max(rank) over
  V4: cat_d5, cat_d6, cat_alt,
      cat_d5_f20, cat_d6_f20, cat_alt_f20,
      cat_d5_r16, cat_d6_r16, cat_alt_r16, cat_alt_r16b
  ∪ opus: merger_ord8, v2_cat_alt8
  ∪ w12:  cat_w12_d5   (8 seeds × 10-fold, fixed trees)
```

| 臂 | bagged OOF | 协议 |
|---|---:|---|
| V4 十臂 | 0.696–0.699 | 固定树，无 ES |
| merger_ord8 | 0.69660 | Ordered，固定 800 树，8 seed |
| v2_cat_alt8 | 0.69704 | Plain alt，固定 800 树，8 seed |
| cat_w12_d5 | **0.69986** | main∪alt 联合 FE，8 seed |

`gap_v5` 在规则表中预登记（`views_max_v4_mag*`），嵌套选择中与 `ma_w12` 接近，最终多数票为 **不含 gap 的 `ma_w12`**。

## 预估公开榜

用已核实诚实锚 gap ≈0.0092–0.0095：

**≈ 0.7130–0.7133**（期望略高于已测冠军 0.71222，**不是保证**）。

Bayes（当前融合 in-sample isotonic）≈0.7075 → 诚实 LB 上限约 **0.715–0.717**。

## 已拒绝（均有同尺子数字）

| 方向 | 结果 |
|---|---|
| mine_noxb_honest 进 max | 拖 full OOF / 嵌套不选 |
| w8 fast | 0.68285 < main fast 0.68961 |
| w12 d6l6 进规则 | nested 0.70377 < 0.70382 |
| w12 f20 进规则 | nested 0.70379 < 0.70382 |
| 8-seed vs 4-seed w12 | 嵌套持平（仅降噪） |
| Ordered×alt / ×w12 | 弱于同种子 Plain |
| Ordered×main | 已有 merger_ord8 |
| PairLogit | OOF ≈0.62–0.65 |
| RMSE 固定树 | bag ≈0.690 << cls |
| 语义 FE 去 ES | OOF ≈0.682 |
| ratio-mid-focus | 弱于 plain |
| iter-jitter | ≈ plain，无增益 |

## 复现

```bash
bash run_v4ext.sh
# 或
PYTHONPATH=src2:src3:src4 python3 src4/fuse4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --report artifacts/v4_ext/fusion_report_v4ext.json
PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --target 0.707 --out artifacts/v4_ext/audit.json
```

臂溯源见 `artifacts/v4_ext/PROVENANCE.md`。
