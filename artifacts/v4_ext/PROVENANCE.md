# V4-ext arm provenance

融合目录：`artifacts/v4/`（`fuse4.py` 只读该目录下的 `arm_*.npz`）。  
副本与报告：`artifacts/v4_ext/`。

| arm 文件 | 来源 | 协议 |
|---|---|---|
| `arm_cat_d5/d6/alt*.npz` 等 V4 十臂 | main V4 / worlds10 & v4 parts | 固定树，无 ES |
| `arm_merger_ord8.npz` | opus `v5_honest` / zcode `merger_ord8.npz` | Ordered CatBoost，固定 800 树，8 seeds，无 ES |
| `arm_v2_cat_alt8.npz` | opus `v5_honest` / zcode | alt 编码，固定 800 树，8 seeds，无 ES |
| `arm_gap_v5.npz` | opus `v5_honest` `arm_gap` | B6 gap 视图，固定树，无 ES（预登记，嵌套未主选） |
| `arm_cat_w12_d5.npz` | 本分支 `run_world.py --world w12 --preset d5`，8 seeds × 10-fold | 固定树，无 ES |

zip 来源分支：`20260808-cursor-opus-grok-glm`、`zcode-v4-max3`。

## 拒绝产物（不进提交规则）

`artifacts/v4_ext/rejected/` 与筛查目录（`v4_rmse_screen`、`v4_semantic_screen`、`v4_ordered_parts` 等）仅作实验记录，**不得**当作交付臂。
