# id 信号演进（v3 → fp_v7）

## fp_v7（当前冠军）

```text
cmean = 0.5*cross-byte OR + 0.5*cross-byte XOR   # same-bit, 选择无关全量池
score = 0.15*v3 + 0.10*bits + 0.05*xs + 0.22*and_all
      + 0.06*tri + 0.06*or + 0.06*xor + 0.30*cmean
```

OOF **0.75838** / nest **0.75851**。复现：`bash run_fp_v7.sh`

详见 `docs/FIRST_PRINCIPLES.md`。

## 历史

| 版本 | OOF | 复现 |
|---|---:|---|
| v3 dual | 0.70717 | `bash run_am40_idbytes.sh` |
| fp_v4 | 0.71640 | `bash run_fp_v4.sh` |
| fp_v5 and_heavy | 0.72880 | `bash run_fp_v5.sh` |
| fp_v6 heavy_xor | 0.74342 | `bash run_fp_v6.sh`（现为 v7 tempered） |
