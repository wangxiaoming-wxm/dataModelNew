# id 信号演进（v3 → fp_v4 → fp_v5）

## fp_v5（当前冠军）

```text
and_all = mean(within-byte bitAND TE pool, cross-byte same-bit AND TE pool)
score   = 0.30*v3_dual + 0.25*bits64 + 0.10*x0_18 + 0.35*and_all
```

OOF **0.72880**。复现：`bash run_fp_v5.sh`

详见 `docs/FIRST_PRINCIPLES.md`。

## 历史

- v3 dual：OOF 0.70717（`bash run_am40_idbytes.sh`）
- fp_v4：OOF 0.71640（`bash run_fp_v4.sh`）
