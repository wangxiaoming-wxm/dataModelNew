# vz20：独立审核后的冲榜接管

保险理赔二分类，指标 ROC-AUC。本目录是**新分支上的独立接管**，不以 rebuild V2 / fp_v* 虚高 OOF 为目标。

## 一句话结论

可迁移天花板仍是已上线的 **W62 = 0.71503**。  
**W62 与 AM40 都不要再交**（用户已交 W62；AM40 测试 Spearman≈0.999）。  
本轮换轴尚未产出「排名不同、且有证据可能超过 0.715」的新票。

## 真实线上校准（主人反馈）

| 提交 | 本地 OOF | 线上 |
|---|---:|---:|
| W62（已交） | 0.70159 | **0.71503** |
| vz17 | 0.70170 | 0.71487 |
| vz19 | 0.70355 | **0.71298** |
| rebuild V2 | 0.69518（诚实 nested） | 未交；弱于 W62 |
| 前三门槛 | — | ~0.72 |
| 第一名 | — | **0.749** |

vz19 的 OOF→线上 +0.013 外推已失效。本地抬升若来自 max2/byteTE/同构融合，会伤线上。

## 文件

| 路径 | 说明 |
|---|---|
| `src/probe_new_axes.py` | 分类/排序/切片专家探针 |
| `src/probe_round2.py` | KNN / YetiRank / 第三世界 |
| `src/probe_round3.py` | Langevin / 分裂准则 / Quantile |
| `src/probe_ordered.py` | Ordered `fold_len_multiplier` / permutation（已证伪） |
| `src/probe_w1.py` | w1=0 弱切片交叉与上采样（已证伪） |
| `src/probe_joint.py` | 双世界拼单模型（弱于 rank 融合） |
| `docs/EVIDENCE.md` | 换轴证据 |
| `docs/INDEPENDENT_AUDIT.md` | 对 rebuild V2 的独立审核 |
| `submission_vz20.csv` | 历史 AM40 克隆，**不要交** |

## 复现探针

```bash
python3 vz20/src/probe_new_axes.py
python3 vz20/src/probe_round3.py
python3 vz20/src/probe_ordered.py
python3 vz20/src/probe_w1.py
python3 vz20/src/probe_joint.py
```

依赖：`artifacts/super714/best_v1_{oof,test}.npy` 与 `data/{train,test,submit_sample}.csv`。

## 提交建议

- **不要交 W62、AM40、rebuild V2、fp_v6–v8。**
- 只有诚实 OOF 相对冻结 W62 **+0.001** 且 test Spearman < 0.995，才生成新的 `submission_vz20.csv`。本轮未达到。
