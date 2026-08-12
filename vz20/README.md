# vz20：独立审核后的可复现交付

保险理赔二分类，指标 ROC-AUC。本目录是**新分支上的独立接管**，不以 rebuild V2 / fp_v* 虚高 OOF 为目标。

## 一句话结论

在诚实协议与真实线上锚点下，当前可迁移天花板仍是 **W62 = 0.71503**。  
vz20 实验主交是同构 **AM40**（本地 OOF **0.70181**，比 W62 +0.00022）。  
**没有证据**支持冲击 0.72（前三）或 0.749（冠军）。

## 真实线上校准（主人反馈）

| 提交 | 本地 OOF | 线上 |
|---|---:|---:|
| W62 | 0.70159 | **0.71503** |
| vz17 | 0.70170 | 0.71487 |
| vz19 | 0.70355 | **0.71298** |
| rebuild V2 | 0.69518（诚实 nested） | 未交；弱于 W62 |
| 前三门槛 | — | ~0.72 |
| 第一名 | — | **0.749** |

vz19 的 OOF→线上 +0.013 外推已失效；多出来的 max2/byteTE 没有变成线上分。

## 文件

| 路径 | 说明 |
|---|---|
| `submission_vz20.csv` | AM40，实验主交 |
| `submission_vz20_w62_anchor.csv` | 已验证线上 0.71503 的保底 |
| `src/build_vz20.py` | 从冻结 best_v1 复现 |
| `docs/INDEPENDENT_AUDIT.md` | 对子 agent V2 方案的独立审核 |
| `docs/EVIDENCE.md` | 本轮换轴实验（均未过门禁） |
| `docs/STATUS.md` | 相对 0.72/0.749 的诚实状态 |

## 复现

```bash
python3 vz20/src/build_vz20.py
# 或
bash run_vz20.sh --verify
```

依赖：仓库内已有 `artifacts/super714/best_v1_{oof,test}.npy` 与 `data/{train,test,submit_sample}.csv`。

## 提交建议

- **要分数、一次名额：交 W62 锚点。**
- **有额外名额：可试 AM40（vz20），期望仍在 0.715 附近，不是夺冠票。**
- 不要交 rebuild V2 / fp_v6–v8。
