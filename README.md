# SUPER714

干净、可复现的 CatBoost 双臂方案。

**当前主提交 = 复现 `real714` / best_v1 冠军**（本地 OOF `0.70128`，线上锚点 `0.71453`）。  
这是对冠军的复现，**不是**对 0.71453 的超越；相对仓库内旧线上成绩（如 v4_max3 `0.71222`）更高。  
仅当完整 TE 第三臂通过预注册门槛时，主提交才会升级为 max3。

## 快速开始

```bash
pip install -r requirements.txt

# 秒级核验预计算 best_v1 产物与主提交
bash run_super714.sh --verify

# 冒烟：完整 TE 通路，2 folds × 1 seed × 1 bag，不覆盖主提交
bash run_super714.sh --smoke

# 完整训练唯一候选 TE 臂：8 seeds × 5 folds × 3 bags
bash run_super714.sh

# 可选：从头重训 best_v1 双臂（默认直接使用已校验冻结产物）
bash run_super714.sh --baseline-only
```

默认数据目录是 `data/`，也可覆盖：

```bash
DATA_DIR=/path/to/data bash run_super714.sh --smoke
```

主提交：`submissions/submission_super714.csv`。完整训练只会在 TE 臂同时通过
`AUC>0.697`、与 main 的 Spearman `<0.90`、max3 增益 `>0.001` 时升级它；
否则自动保持 best_v1 max2。
方案与诚实门槛见 [`docs/SOLUTION.md`](docs/SOLUTION.md)。

## 目录

```text
data/                 train.csv / test.csv / submit_sample.csv
real714_pkg/          解压后的只读参考包；运行不依赖它
src_super/            SUPER714 训练、特征与验收代码
artifacts/super714/   已验证锚点及训练产物
submissions/          提交文件
docs/                 方案、策略、复盘与协作记录
run_super714.sh       一键入口
```

约束：只使用给定数据；无外部数据、无伪标签、无测试标签；目标编码必须折内生成。
