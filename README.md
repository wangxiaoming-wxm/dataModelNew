# 车险索赔预测 — task-20260808-cursor

> 提交文件：`submissions/submission_v2.csv`；本地嵌套 OOF AUC **0.69856**（诚实协议，见 `docs/RESULTS.md`）。

赛题：根据保单与车辆等多维信息，预测投保人未来一年内是否发生索赔。评价指标 ROC-AUC。
数据在 `data/`（`train.csv` 14930 行含 `label`，`test.csv` 6398 行，`submit_sample.csv` 为提交模板）。

本分支是**重写的一版方案**，不是在 B7 上做增量调参。核心是先把数据的生成机制逆向清楚，
再据此重建特征与集成方式，并在一个**不在验证折上早停**的诚实协议下评估。

| 项 | 内容 |
|---|---|
| 提交文件 | `submissions/submission_v2.csv` |
| 本地口径 | 重复分层 5 折，固定树数，无验证折早停，融合规则嵌套选择 |
| 对照 | B7 提交（`submissions/submission_b7_closest_honest.csv`）公开榜 0.70722 |

结果数字见 [`docs/RESULTS.md`](docs/RESULTS.md)。

---

## 关键发现

**1. 44 个特征里有 27 列是匿名化噪声，对标签的预测力严格为零。**

`source`（11 类）是隐藏的车型 ID；`V`、`x19`、`code`、`t3`、`cc`、`max_g`、`x0`–`x17`
都是它的确定性函数加**均匀噪声**（组内残差宽度精确等于 `极差/√12`）。
`livability` ≡ `region`，`x20` ≈ `1.2·condition − 0.53 + U(−1.5, 1.5)`，`x18` 是无条件噪声。
把这些残差单独或联合拿去预测标签，全部落在随机置换的噪声带内。

**2. `condition` 必须先按车型归一化，之后 `days/condition` 是全数据最强的排序变量。**

各车型的 `condition` 量纲差 3 倍以上，原始列把车型差异和个体差异混在一起。
`source × condition` 是所有二阶交叉里增益最大的一项（+0.0292）。归一化并与 `days` 组合后：

| 特征 | 单列 AUC |
|---|---:|
| `days`（原始最强列） | 0.593 |
| `condition` | 0.532 |
| `condition / median(condition \| source)` | 0.567 |
| `days / (condition / median(condition \| source))` | **0.620** |

**3. 反直觉的一点：噪声列仍然值得放进 CatBoost，但原因不是它们有信息。**

它们是同一批真实交互的不同扰动离散化，CatBoost 对每个类别列都算有序目标统计量，
多个扰动版本并存等于让树集成对同一统计量做了平均。加回去后单模型 OOF 从 0.6793 升到 0.6892。
`src2/jitter.py` 把这个机制变成可控的：用行 `id` 哈希造确定性扰动流，每个种子一套不同的重编码。
LightGBM 与 GLM 臂里这些列全部剔除——它们没有有序目标统计量保护自己。

**4. B7 的本地分被高估了。** `src/insurance_claim/train_b6.py:321` 把外层验证折同时用于
早停和产生 OOF。同一份特征、同一组参数下实测偏差约 +0.0015（bag 口径 0.69111 → 0.68948）。
数值不大，但足以让"本地分"不能作为决策依据，所以本分支所有对比都改用固定树数。

细节见 [`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md)。

---

## 复现

```bash
python3 -m pip install -r requirements.txt

# 主视图，两段各 4 个种子（抖动流互不重叠）
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20260 20261 20262 20263 --stream-base 0  --out artifacts/v2a
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20264 20265 20266 20267 --stream-base 10 --out artifacts/v2b

# 第二套编码世界
PYTHONPATH=src2 python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20280 20281 20282 20283 --stream-base 0  --out artifacts/v2alt
PYTHONPATH=src2 python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20284 20285 20286 20287 --stream-base 10 --out artifacts/v2alt2

# 沿用上一版 gap 视图，但改用诚实协议重跑，只为多样性
python3 src2/run_gap_arm.py --seeds 20290 20291 20292 20293 --out artifacts/v2gap

# 合并 → 融合 → 写提交
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2a artifacts/v2b --out artifacts/v2main
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2alt artifacts/v2alt2 --out artifacts/v2altmerged
PYTHONPATH=src2 python3 src2/collect.py --out artifacts/v2
PYTHONPATH=src2 python3 src2/fuse.py --dir artifacts/v2 --submission submissions/submission_v2.csv
```

## 目录

```text
data/                比赛数据
src2/                本分支的方案
  features.py        特征工程（两套编码世界）
  jitter.py          确定性扰动重编码
  te.py              折内嵌套目标编码
  arms.py            四个模型臂 + 重复 CV 运行器
  run_oof.py         主/备视图的 OOF 与 test 预测
  run_gap_arm.py     上一版 gap 视图，改用诚实协议
  merge_runs.py      合并多段运行
  fuse.py            预登记规则 + 嵌套选择 + 写提交
eda/                 数据结构逆向的探查脚本
exp/                 特征与超参的对照实验
docs/
  DATA_STRUCTURE.md  数据生成机制逆向（本方案的地基）
  SOLUTION.md        方案与协议
  RESULTS.md         结果与诚实性检查
src/                 上一版 B7 代码（仅用于对照与 gap 视图）
docs/B7_*.md         上一版的文档，保留作历史记录；其中的本地分数用的是旧口径，不要与本版直接比较
docs/supervision/    同上
```
