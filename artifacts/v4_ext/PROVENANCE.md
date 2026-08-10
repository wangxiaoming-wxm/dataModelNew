# V4-ext arm provenance

| arm file | source | protocol |
|---|---|---|
| arm_merger_ord8.npz | opus `v5_honest` / zcode `merger_ord8.npz` | Ordered CatBoost, fixed 800 trees, 8 seeds, no ES |
| arm_v2_cat_alt8.npz | opus `v5_honest` / zcode | alt encoding, fixed 800 trees, 8 seeds, no ES |
| arm_gap_v5.npz | opus `v5_honest` arm_gap | B6 gap view, fixed 1000 trees, 4–8 seeds, no ES |

Copied into `artifacts/v4/` for fuse4 single-dir load. Original bytes from zip packages on branches
`20260808-cursor-opus-grok-glm` / `zcode-v4-max3`.
