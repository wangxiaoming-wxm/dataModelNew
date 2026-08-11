STATUS: DONE

# ML 工程交付状态

- 主提交：`submissions/submission_super714.csv` = 已验证 best_v1 max2（线上锚点 **0.71453**）
- 预计算锚点：`artifacts/super714/best_v1_{oof,test}.npy`（SHA 固化，秒级 `bash run_super714.sh --verify`）
- TE 第三臂：已实现双层诚实 FE + 训练入口；**smoke 未过门槛，拒绝覆盖主提交**（见 `docs/TE_GATE_RESULT.md`）
- 训练入口：`bash run_super714.sh` / `--smoke` / `--verify` / `--baseline-only`
- 方案文档：`docs/SOLUTION.md`
- 数据挖掘交接：`docs/HANDOFF_TO_ML.md` / `docs/ARM_SELECTION.md` / `docs/SCHEME_INVENTORY.md`

相对仓库内其余已验证线上方案（v4_max3 0.71222 等），本交付为当前最强可复现方案。
