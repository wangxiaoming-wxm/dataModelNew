# id 信号演进（v3 → fp_v4）

## v3 dual（已被 fp_v4 吸收）

```text
V7 = [b3, b2hi, b1hi, b0, b4hi, b5, b7, b7hi, b5hi, b6hi]
V2 = [b0, b4, b5, b7, b2hi, p47]
id_pool = 0.70 * pool(V7) + 0.30 * pool(V2)
v3      = 0.55 * AM40 + 0.45 * id_pool
```

OOF **0.70717**。复现：`bash run_am40_idbytes.sh`

## fp_v4（当前冠军）

在 v3 之上加入**选择无关**的更细 id 与回收列：

```text
bits = mean_rank(TE(all 64 bit planes))
xs   = mean_rank(TE(qbin20(x0..x18)))   # 模型曾忽略的原始列
score = 0.50 * v3 + 0.35 * bits + 0.15 * xs
```

OOF **0.71640**。复现：`bash run_fp_v4.sh`

详见 `docs/FIRST_PRINCIPLES.md`。
