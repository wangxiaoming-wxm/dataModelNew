STATUS: DONE_BASELINE; TE_FULL_PENDING

# ML 工程交付状态

- 主提交：`submissions/submission_super714.csv` = **复现** best_v1 max2  
  - 本地 OOF **0.70128**  
  - 线上锚点 **0.71453**  
  - **不是对 0.71453 的超越**；相对旧方案（v4_max3 0.71222 等）为更强
- 秒级验收：`bash run_super714.sh --verify`（校验冻结产物 SHA + 提交等于 fuse clip）
- TE 候选：已实现；smoke 未过门槛且不覆盖主提交；完整 8×5×3 见 `logs/super714_full_te.log` / 将来的 `metrics.json`
- 协议口径：`docs/PROTOCOL.md`
- 方案：`docs/SOLUTION.md`
