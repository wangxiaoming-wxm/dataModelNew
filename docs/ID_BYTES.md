# id 字节信号 → AM40+idbytes

## 能不能用？

**能。** `id` 是原始字段；解析 hex 字节做 fold-local TE，不用 test 标签、不用外部数据，与用 `days`/`region` 同类。

## 复现结果（本仓库）

| byte | fold-local TE AUC | \|sig\| |
|---|---:|---:|
| 0 | 0.5248 | **0.0248** ★ |
| 4 | 0.4860 | **0.0140** ★ |
| 5 | 0.4857 | **0.0143** ★ |
| 7 | 0.5221 | **0.0221** ★ |

- 直接把 id 字节当类别特征塞进 CatBoost：**不涨**（strong Δ≈0；全字节 Δ≈−0.0027）  
- 原因：单变量有信号，但被现有树特征吸收  
- **正确用法**：fold-local TE → 与 AM40 **线性混合**（id 池与 AM40 Spearman≈0）

## 融合

```text
id_pool = mean_rank( flip_if_auc<0.5( TE_5fold(id_byte_b) ) for b in [0,4,5,7] )
score   = 0.75 * AM40 + 0.25 * id_pool
```

| 指标 | 值 |
|---|---|
| AM40 OOF | 0.701811 |
| **AM40+idbytes OOF** | **0.704413** |
| Δ | **+0.002601** |
| nested OOF mean | 0.704082 |
| 提交 | `submissions/submission_am40_idbytes.csv` |

```bash
bash run_am40_idbytes.sh
bash run_am40_idbytes.sh --verify
```

## 提交建议

本地增益远大于 Bags/Plus/region；与 W62 Spearman≈0.986（有差异）。  
次数紧时：这是目前**最值得搏一次线上**的新文件（仍有 OOF↔LB 落差风险，但方向比同构调参靠谱）。
