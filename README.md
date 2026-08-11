# 车险索赔预测 — V4-ext（本分支当前交付）

> **提交文件：`submissions/submission_v4ext.csv`**  
> 诚实嵌套 OOF（20 block seed 均值）：**0.70381**  
> 对照 V4：0.70303 → **+0.00079**；对照 opus `v5_honest` 本地 0.70253 → **+0.00128**  
> 预估公开榜：**约 0.7130–0.7133**（按已核实诚实 gap 校准，**非承诺**）  
> 诚实性门：`honesty_passed=true`；目标 0.725：**不能诚实承诺**

赛题：根据保单与车辆等信息预测一年内是否索赔，指标 ROC-AUC。  
数据在 `data/`（`train.csv` 14930 行含 `label`，`test.csv` 6398 行）。

本分支在 **main V4 诚实协议**上并入 opus/zcode 已验证诚实臂，并加入通过闸门的 `w12`（main∪alt 联合特征）臂，用预登记 `max(rank)` 规则做嵌套选择。

| 项 | 内容 |
|---|---|
| 提交文件 | `submissions/submission_v4ext.csv` |
| 提交规则 | `views_max_v4_ma_w12` |
| 主报口径 | 20× block-seed 嵌套选择均值 |
| 协议 | 固定树数、无验证折早停、label-free FE、不碰测试标签 |
| 一键复现 | `bash run_v4ext.sh` |

## 先读文档

1. [`docs/DELIVERY.md`](docs/DELIVERY.md) — **交付清单与路径（从这里开始）**  
2. [`docs/SUBMISSION_AUDIT.md`](docs/SUBMISSION_AUDIT.md) — 独立过拟合审核与提交排序  
3. [`docs/V4EXT.md`](docs/V4EXT.md) — 配方、数字、已拒方向  
4. [`docs/STATUS_HONEST_PUSH.md`](docs/STATUS_HONEST_PUSH.md) — 推高过程与 0.725 终裁  
5. [`docs/LB_BOARD.md`](docs/LB_BOARD.md) — 用户核实的公开榜板  

历史对照（只读）：[`docs/V4.md`](docs/V4.md)、[`docs/V3.md`](docs/V3.md)、[`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md)。

## 复现（秒级）

```bash
python3 -m pip install -r requirements.txt
bash run_v4ext.sh
```

应看到：`nested_oof_mean ≈ 0.70381`，`submitted_rule = views_max_v4_ma_w12`，`honesty_passed = true`。

## 目录（交付相关）

```text
submissions/submission_v4ext.csv   ★ 正式提交
run_v4ext.sh                       ★ 融合 + 监督
src4/fuse4.py                      预登记规则（含 V4-ext / w12）
src4/audit_v4.py                   独立监督（与 fuse4 共用 RULES）
artifacts/v4/                       融合用全部 arm_*.npz
artifacts/v4_ext/                   报告、溯源、审计 JSON
docs/DELIVERY.md                   交付说明
```
