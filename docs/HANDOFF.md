# 交接说明

写给下一个接手继续开发和训练的人。读完这一份 + `docs/DATA_STRUCTURE.md` 就能上手。

---

## 0. 现在处于什么状态

| 项 | 值 |
|---|---|
| 提交文件 | `submissions/submission_v2.csv` |
| **本地诚实 AUC（主报）** | **0.69856**（嵌套选择 OOF） |
| 同规则全量 OOF | 0.69910 |
| 采用的融合规则 | `views_max` = `max(rank(cat_d5), rank(cat_d6), rank(cat_alt))` |
| 对照：B6 `gap` 视图同协议重跑 | 0.69044 |
| 对照：B7 旧口径本地分 | 0.702705（**不可比**，含 +0.0025 早停乐观，且口径不同） |

所有中间产物都在仓库里，**不需要重跑就能复现融合和提交**：

```bash
PYTHONPATH=src2 python3 src2/fuse.py --dir artifacts/v2 --submission submissions/submission_v2.csv
```

---

## 1. 环境

Python ≥ 3.10。CPU 即可（本项目全程在 4 核机器上跑完）。

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` 里 `lightgbm` / `xgboost` 只被次要臂和早期实验用到，主力只需要 `catboost`。

---

## 2. 目录导航

```text
data/                    比赛数据（SHA256 见 data/SHA256SUMS.txt，verify.py 会校验）

src2/                    ★ 本方案的全部代码
  features.py            三套编码世界的特征工程 + 主办方噪声视图
  jitter.py              确定性抖动重编码（按行 id 哈希，train/test 一致）
  te.py                  折内嵌套目标编码（供 LightGBM / GLM 臂使用）
  arms.py                模型臂定义 + 三种特征帧的构造入口
  run_oof.py             ★ 训练入口：跑一个视图的若干种子，产出 OOF 与 test 预测
  run_gap_arm.py         用诚实协议重跑上一版 B6 gap 视图（只为多样性与对照）
  merge_runs.py          把多段 run_oof 的产物合并成更大的 bag
  collect.py             把各视图的臂文件汇总到 artifacts/v2
  fuse.py                ★ 预登记规则 + 嵌套选择 + 写提交
  verify.py              ★ 诚实性检查（数据校验和 / 打乱标签对照 / 提交格式）
  common.py              早期实验用的共享 CV 工具（现在只被 exp/ 用）

artifacts/
  v2/                    ★ 最终用于融合的臂文件 + fusion_report.json + verify.json + es_bias.json
  v2a v2b v2c            主视图三段运行（各 4 个种子）
  v2alt v2alt2 v2alt3    第二编码世界三段运行
  v2alt2w v2alt2w_b      第三编码世界两段运行
  v2main v2altmerged v2alt2merged   合并后的中间结果
  v2gap                  gap 视图
  b6_frozen b7_*         上一版 B7 的冻结产物（仅对照）

logs/training/           ★ 每一次训练运行的原始日志（含逐种子 OOF 与耗时）
exp/                     筛选实验脚本（exp01–exp15），结论汇总在 docs/EXPERIMENTS.md
eda/                     数据结构逆向的探查脚本
docs/                    见下

src/                     上一版 B7 代码，保留用于对照和提供 gap 视图
```

文档：

建议的阅读顺序：

| 顺序 | 文件 | 内容 |
|---:|---|---|
| 1 | `docs/DATA_STRUCTURE.md` | 数据生成机制逆向，整个方案的地基 |
| 2 | `docs/STRATEGY.md` | 解题思路、架构、避开的坑、冲榜路线图与两套备用方案 |
| 3 | `docs/EXPERIMENTS.md` | 15 组对照实验的台账，含**明确无效的方向清单** |
| 4 | `docs/RESULTS.md` | 各臂 / 融合 / 诚实性检查的最终数字 |
| 5 | `docs/HANDOFF.md` | 本文件：环境、重跑、协议红线 |
| — | `docs/legacy/` | 上一版（B5/B6/V10/B7）的文档。**本地分数是旧口径，不可直接比较**，
先读 `docs/legacy/README.md` 里的口径说明和已被推翻的结论 |

---

## 3. 从零重跑（约 3.5 小时 @ 4 核）

一条命令：

```bash
bash run_all.sh
```

它按顺序做的事（也可以单独跑其中任意一步）：

```bash
# 主编码世界，三段各 4 个种子；--stream-base 让每段用不同的抖动流
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20260 20261 20262 20263 --stream-base 0  --out artifacts/v2a   # ~33 min
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20264 20265 20266 20267 --stream-base 10 --out artifacts/v2b   # ~35 min
PYTHONPATH=src2 python3 src2/run_oof.py --arms cat_d5 cat_d6 --seeds 20268 20269 20270 20271 --stream-base 20 --out artifacts/v2c  # ~32 min

# 第二编码世界，三段各 4 个种子
PYTHONPATH=src2 python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20280 20281 20282 20283 --stream-base 0  --out artifacts/v2alt   # ~12 min
PYTHONPATH=src2 python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20284 20285 20286 20287 --stream-base 10 --out artifacts/v2alt2  # ~18 min
PYTHONPATH=src2 python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20288 20289 20294 20295 --stream-base 20 --out artifacts/v2alt3  # ~12 min

# 第三编码世界（当前没进最终融合，保留以便继续改进）
PYTHONPATH=src2 python3 src2/run_oof.py --view alt2 --arms cat_alt2 --seeds 20300 20301 20302 20303 --stream-base 0  --out artifacts/v2alt2w    # ~14 min
PYTHONPATH=src2 python3 src2/run_oof.py --view alt2 --arms cat_alt2 --seeds 20304 20305 20306 20307 --stream-base 10 --out artifacts/v2alt2w_b # ~14 min

# 上一版 gap 视图，诚实协议
python3 src2/run_gap_arm.py --seeds 20290 20291 20292 20293 --out artifacts/v2gap   # ~19 min

# 合并 → 汇总 → 融合 → 写提交 → 校验
# 两遍：第一遍保住 8 种子的 lgb_te / glm（v2c 只跑了 CatBoost 臂），
# 第二遍用 12 种子的结果覆盖 cat_d5 / cat_d6
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2a artifacts/v2b --out artifacts/v2main
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2a artifacts/v2b artifacts/v2c --out artifacts/v2main
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2alt artifacts/v2alt2 artifacts/v2alt3 --out artifacts/v2altmerged
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2alt2w artifacts/v2alt2w_b --out artifacts/v2alt2merged
PYTHONPATH=src2 python3 src2/collect.py --out artifacts/v2
PYTHONPATH=src2 python3 src2/fuse.py --dir artifacts/v2 --submission submissions/submission_v2.csv
PYTHONPATH=src2 python3 src2/verify.py
```

分段跑而不是一次跑 12 个种子，是为了每段能用不同的抖动流（`--stream-base`），
同时也方便断点续跑——每段产物独立，`merge_runs.py` 会按等权平均合并（每段模型数相同时是精确的池化）。

`run_oof.py` 的耗时几乎全在 CatBoost 上，特征帧构造只要 1 秒。
`cat_d5` 约 260 秒/种子，`cat_d6` 约 230 秒/种子，`cat_alt` 约 180 秒/种子，`lgb_te`/`glm` 各 3 秒/种子。

---

## 4. 必须守住的协议红线

改任何东西之前先看这一节。这些是本方案能自证清白的基础：

1. **不许在外层验证折上早停或做任何选择。** 树数在开跑前定死。
   （上一版就栽在这里，实测乐观 +0.0025，见 `artifacts/v2/es_bias.json`。）
2. **特征工程不许碰标签。** 分位切点、每个车型的 condition 量纲、频次编码、抖动流
   全部是 label-free 的，所以在 train+test 上拟合一次是安全的。
   加新特征时如果用到了 `y`，必须走 `src2/te.py` 的折内嵌套编码。
3. **融合规则必须先登记再看分。** 规则集写在 `src2/fuse.py` 的 `RULES` 里，
   最终由嵌套选择决定，主报数字是 `nested_oof_auc`。
   加规则会增加选择方差，不要为了刷 0.001 往里塞十几条。
4. **每次改完跑一遍 `src2/verify.py`。** 特别是打乱标签对照——
   它会把整条管线在随机置换的标签上重跑，AUC 必须回落到 0.5 附近（当前 0.4983）。
   只要新加的变换偷看了标签，这一条立刻会红。
5. **不要拿榜单分当调参反馈。** 公开榜的抽样标准误约 0.011–0.016，
   照着它调等于在拟合噪声，本地口径会先崩。

---

## 5. 接下来最值得做的事（摘要）

完整的路线图、每一项的预期收益 / 成本 / 风险 / 验证方式，以及两套备用方案，
都在 **[`docs/STRATEGY.md`](STRATEGY.md)**。这里只留一页纸的摘要，避免两处内容分叉。

要夺冠还差 **+0.007 ~ +0.012** 的本地诚实分（目标 0.706 ~ 0.710）。按性价比排序：

| 优先级 | 动作 | 预期收益 | 机时 |
|---|---|---:|---:|
| P0 | 5 折换 10 折（学习曲线已支持，见 STRATEGY §5.3） | +0.003 ~ +0.008 | +80% |
| P1 | 再造 2–3 个「编码世界」，每个必须 ≥ 0.694 | +0.003 ~ +0.006 | 每个约 30 min |
| P2 | 把 `lgb_te` / `glm` 做强到 0.688 以上（它们最解耦） | +0.002 ~ +0.005 | 几乎为零 |
| P3 | 树数随机化，制造合法的 bagging 多样性 | +0.002 | 35 min |
| P4 | 折内多模型各看一套抖动再平均 | +0.001 ~ +0.002 | 线性 |

一句话原则：**这份数据上真正在起作用的机制是「对编码做平均」**——
能增加编码多样性的动作有回报，换模型族、加种子、调超参都没有。

### 不要再做的事

见 [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) 第 7 节的无效方向清单。特别是：
别再去挖 `x0`–`x18` / `cc` / `max_g` / `V` 的残差（纯噪声），
别再盲目扩交叉数量，别再调 `max_ctr_complexity`，别再指望换模型族翻盘。

---

## 6. 关于榜单分的现实预期

本地 0.699 不等于榜单 0.716。公开榜是测试集的一个子集，
6398 行、约 640 个正例时 AUC 的抽样标准误约 0.011，只用一半数据约 0.016。

本方案相对 B7 的 **+0.008 本地增益是可信、可复现的那部分**，
落到公开榜上会再叠加一个 ±0.01 量级的随机项。合理预期 **0.711–0.719**，中位数约 0.715。

差距的完整拆解（测试端集成更强 / 子集抽样 / 无分布偏移）和"要夺冠需要本地到多少"的推算，
见 [`docs/STRATEGY.md`](STRATEGY.md) 第 1 节。
