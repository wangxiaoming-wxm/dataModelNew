STATUS: DONE

# ML 工程交付状态

- 主提交：`submissions/submission_super714.csv` = **复现** best_v1 max2  
  - 本地 OOF **0.70128** / 线上锚点 **0.71453**  
  - **不是对 0.71453 的超越**；相对旧方案（v4_max3 0.71222 等）更强
- 完整 TE 门槛：`artifacts/super714/metrics.json`  
  - TE AUC 0.69964 ✅ / Spearman(main) 0.99767 ❌ / max3 Δ −0.00018 ❌  
  - **已拒绝**，不覆盖主提交（见 `docs/TE_GATE_RESULT.md`）
- 秒级验收：`bash run_super714.sh --verify`
- 协议：`docs/PROTOCOL.md`
