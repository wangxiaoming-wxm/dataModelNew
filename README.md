# SUPER714 — 车险索赔预测（可复现交付）

| 项 | 值 |
|---|---|
| 主提交 | [`submissions/submission_super714.csv`](submissions/submission_super714.csv) |
| 本地 OOF（max2） | **0.70128** |
| 线上锚点 AUC | **0.71453** |
| 方案 | best_v1 双臂 CatBoost：cond_r×Ordered + rate×Plain，`max(rank)` 融合 |

这是对已验证线上冠军 **best_v1** 的干净复现（不是对 0.71453 的再超越）。  
相对历史方案（如 v4_max3 ≈ 0.71222）更高。完整 TE 第三臂已评估并**拒绝**（与主臂共线），详见 [`docs/TE_GATE_RESULT.md`](docs/TE_GATE_RESULT.md)。

---

## 60 秒复现（推荐）

```bash
git clone https://github.com/wangxiaoming-wxm/dataModelNew.git
cd dataModelNew
git checkout real714

python3 -m pip install -r requirements.txt
bash run_super714.sh --verify
```

预期输出包含：

```text
PASS: SUPER714 预计算锚点与主提交验收通过
OOF AUC: {'alt': '0.69770', 'fuse': '0.70128', 'main': '0.69992'}
```

可选测试：

```bash
python3 -m unittest discover -s tests -v
```

更完整说明见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

---

## 训练入口（可选）

| 命令 | 作用 | 耗时量级 |
|---|---|---|
| `bash run_super714.sh --verify` | 核验冻结产物 + 主提交（**交付验收**） | 秒级 |
| `bash run_super714.sh --smoke` | TE 通路冒烟，不覆盖主提交 | ~2 分钟 |
| `bash run_super714.sh` | 完整重训 TE 候选臂并按门槛决定是否升级 | ~1.5 小时 |
| `bash run_super714.sh --baseline-only` | 从头重训 best_v1 双臂 | ~1.5–2.5 小时 |

数据默认读 `data/`；也可：

```bash
DATA_DIR=/path/to/dir bash run_super714.sh --verify
# 或
bash run_super714.sh --verify --data-dir /path/to/dir
```

---

## 目录

```text
data/                      比赛数据（train / test / submit_sample）
src_super/                 训练、TE 特征、秒级验收
artifacts/super714/        冻结 OOF/test、manifest、TE 门槛结果
submissions/               主提交 CSV
tests/                     TE 防泄漏单元测试
docs/                      方案与复现文档（见 docs/README.md）
real714_pkg/               只读参考包（线上冠军源材料；运行不依赖）
run_super714.sh            一键入口
requirements.txt           依赖
```

---

## 配方摘要

| 臂 | 特征世界 | 损失 | Boosting | depth | iter | l2 | 种子 |
|---|---|---|---|---:|---:|---:|---|
| main | cond_r / ratio 归一化 | RMSE | Ordered | 5 | 800 | 10 | 2026–2033 × 3 bag |
| alt | rate = days×(1−rank\|source) | RMSE | Plain | 6 | 800 | 6 | 同上 |
| 融合 | `max(rank(main), rank(alt))` | | | | | | |

协议与约束：[`docs/PROTOCOL.md`](docs/PROTOCOL.md)  
方案细节：[`docs/SOLUTION.md`](docs/SOLUTION.md)
