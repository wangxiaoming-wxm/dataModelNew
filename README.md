# SUPER714

干净、可复现的 CatBoost 双臂方案。当前主提交以已验证的 best_v1 为安全基线：
本地 OOF AUC `0.70128`，线上锚点 `0.71453`。SUPER714 仅在第三臂通过预设门槛时才会替换该基线。

## 快速开始

```bash
pip install -r requirements.txt

# 秒级核验预计算 best_v1 产物与主提交
bash run_super714.sh --verify

# 冒烟：2 folds × 1 seed × 1 bag，不覆盖主提交
bash run_super714.sh --smoke

# 完整：8 seeds × 5 folds × 3 bags
bash run_super714.sh
```

默认数据目录是 `data/`，也可覆盖：

```bash
DATA_DIR=/path/to/data bash run_super714.sh --smoke
```

主提交：`submissions/submission_super714.csv`。
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
