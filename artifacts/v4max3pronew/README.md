# artifacts/v4max3pronew

## Official

- `semantic_rmse.npz` — 715 faithful port (5×5×10 RMSE), pooled OOF 0.69597
- `recipe_report.json` / `status_report.json` — fusion admission
- paired `part_semantic_rmse_s*.npz` for bit-rebuild of the pooled arm

## Explored but not in final recipe

- `semantic_logloss.npz` — same FE + Logloss 5×5×10; adding it to max(pro, …)
  **lowers** nested (0.70521 < 0.70557). Kept for audit, not fused into the CSV.

Rebuild CSV:

```bash
python3 src4/build_submission_v4max3pronew.py --check
```
