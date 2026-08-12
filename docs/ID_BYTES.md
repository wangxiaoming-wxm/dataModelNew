# id 信号演进（v3 → fp_v8）

## fp_v8（当前冠军）

```text
bytepair_mean = 0.5*pool(byteXOR) + 0.5*pool(byteAND)
score = 0.80*v7_cross30 + 0.20*bytepair_mean
```

OOF **0.77016** / nest **0.77033**。复现：`bash run_fp_v8.sh`

详见 `docs/FIRST_PRINCIPLES.md`。

## 历史

| 版本 | OOF | 复现 |
|---|---:|---|
| v3 dual | 0.70717 | `bash run_am40_idbytes.sh` |
| fp_v4 | 0.71640 | `bash run_fp_v4.sh` |
| fp_v5 and_heavy | 0.72880 | `bash run_fp_v5.sh` |
| fp_v6 heavy_xor | 0.74342 | `bash run_fp_v6.sh` |
| fp_v7 cross30 | 0.75838 | `bash run_fp_v7.sh`（现为 v8 tempered） |
