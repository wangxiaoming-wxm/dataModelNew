# 交付清单（V4-ext）

## 仓库 / 分支 / 提交文件

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/wangxiaoming-wxm/dataModelNew |
| 分支 | `cursor/honest-push-v4ext-6de7` |
| PR | https://github.com/wangxiaoming-wxm/dataModelNew/pull/9 |
| **正式提交 CSV** | **`submissions/submission_v4ext.csv`** |
| 不要交 | `artifacts/v4_ext/submission_v4ext.csv`（旧快照）、`submission_v4ext_w12.csv`（与正式文件相同，已删除重复） |

## 方案一句话

**main V4 诚实十臂** + opus 诚实臂（`merger_ord8`、`v2_cat_alt8`）+ **`cat_w12_d5`（8 seed）**，预登记规则下嵌套选择得到 `views_max_v4_ma_w12`。

不是 `v4_max3`（后者含早停臂）。

## 关键数字（可复算）

| 口径 | 值 |
|---|---:|
| 诚实 nested（20 seed） | **0.70381** |
| 提交规则全量 OOF | ≈0.70430 |
| 选择乐观（full−nested） | ≈0.00048 |
| `honesty_passed` | true |
| 预估公开榜 | **0.7130–0.7133**（非承诺） |
| 现公开榜冠军（已测） | max3 = **0.71222** |

复现：

```bash
bash run_v4ext.sh
```

## 文档地图

| 文档 | 用途 |
|---|---|
| `docs/SUBMISSION_AUDIT.md` | 过拟合审核 + 提交排序 |
| `docs/V4EXT.md` | 配方、臂表、已拒实验 |
| `docs/STATUS_HONEST_PUSH.md` | 推高过程与 0.725 不可达终裁 |
| `docs/LB_BOARD.md` | 用户核实的全量公开榜 |
| `docs/PLAN_AUDIT.md` | 对规划方案的审核意见 |
| `artifacts/v4_ext/PROVENANCE.md` | 臂文件来源 |
| `artifacts/v4_ext/fusion_report_v4ext.json` | 融合报告 |
| `artifacts/v4_ext/audit.json` | 监督者报告 |

## 公开榜校准锚（用户回执）

| 文件 | LB |
|---|---:|
| submission_v4_max3.csv | 0.71222 |
| submission_v5_honest.csv | 0.71207 |
| submission_v3_max3.csv | 0.71184 |
| submission_v4_honest.csv | 0.71104 |
| submission_v3.csv | 0.71064 |
| submission_v5.csv | 0.71035 |
| submission_v2.csv | 0.70878 |
