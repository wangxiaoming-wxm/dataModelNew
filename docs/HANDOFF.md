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

| 文件 | 内容 |
|---|---|
| `docs/DATA_STRUCTURE.md` | **先读这个。**数据生成机制逆向，本方案的地基 |
| `docs/SOLUTION.md` | 方案与协议 |
| `docs/RESULTS.md` | 各臂 / 融合 / 诚实性检查的最终数字 |
| `docs/EXPERIMENTS.md` | **第二个读这个。**所有对照实验，含明确无效的方向清单 |
| `docs/HANDOFF.md` | 本文件 |
| `docs/B7_*.md`、`docs/b6/`、`docs/supervision/` | 上一版的历史文档，**本地分数用的是旧口径，不要与本版直接比较** |

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

## 5. 接下来最值得做的事（按性价比排序）

### 5.1 做第四、第五套"编码世界"（最高优先级）

本数据上真正在起作用的机制是**对编码做平均**，不是换模型族、也不是加种子：

- 只改 CatBoost 深度：相关 0.997，几乎无增益；
- 换编码世界（`cat_alt`）：相关 0.954，把融合从 0.69539 抬到 0.69910；
- 换模型族（LightGBM / GLM）：太弱，负贡献；
- 加种子：8 → 12 已经饱和。

所以下一步就是照着 `features.build_alt` 的样子再写第三、第四种表达方式。
关键是**新世界必须和现有三个臂同等强度**（bag 后 ≥ 0.694）——
我做的 `cat_alt2` 解耦得很好（相关 0.950）但弱了 0.004，
放进 `max` 反而把 0.69910 拉到 0.69837，因为 `max` 对弱臂敏感（弱臂偶尔会把某些行顶到高位）。

怎么造新世界（都要保住 `cond_r` / `ratio` 这两个核心信号，只换表达方式）：

- 换归一化基准：现在用的是「按 source 取中位数比值」和「按 source 取秩」，
  还可以试「按 source 取分位数映射到正态」「按 source × age 分组」「用稳健 z（中位数 / MAD）」；
- 换分箱：现在是 (5,10,20,40)、(7,13,25)、(4,9,16)，可以试等宽箱、树诱导箱、
  或者对 `ratio` 用对数等距箱；
- 换交叉清单：保住 `condition × source` 这一族，其余重新挑；
- 换抖动参数：`n_views`、`n_bins`、扰动幅度、`n_sub`。

如果 `cat_alt2` 能被调强到 0.694 以上，四臂 `max` 大概率能再上 0.002–0.003。

### 5.2 让弱臂变强，而不是丢掉

`lgb_te`（0.671）和 `glm`（0.665）与主臂的相关只有 0.89–0.93，是全场最解耦的，
可惜太弱所以进融合是负贡献。如果能把它们做到 0.688 以上，
它们带来的增益会比再加一个 CatBoost 世界更大。方向：

- `glm` 现在只有 5 个样条项 + 目标编码 logit，可以加张量交互
  （`source` × `log_cond_r` 的样条）、按 `region` 分层的随机效应；
- `lgb_te` 的目标编码平滑系数、编码列清单都没调过。

### 5.3 复核一处没对齐的地方

B7 报告里 gap 臂 8 种子是 0.69868，减去实测的 +0.0025 偏差后是 0.6962，
仍比我诚实重跑的 gap 臂（4 种子 0.69044，外推 8 种子约 0.692）高约 0.004。
可能的原因：种子数、按概率平均而非按秩平均、以及早停让每个模型树数不同从而
给 bagging 额外增加了多样性。**最后这一条如果成立是有价值的**——
可以在不偷看验证折的前提下复现：给每个模型随机抽一个树数（比如从 300–900 均匀抽），
人为制造树数多样性。这个我没来得及验证，值得试。

### 5.4 不要再做的事

见 `docs/EXPERIMENTS.md` 第 7 节的无效方向清单。特别是：
别再去挖 `x0`–`x18` / `cc` / `max_g` / `V` 的残差（纯噪声），
别再盲目扩交叉数量，别再调 `max_ctr_complexity`，别再指望换模型族翻盘。

---

## 6. 关于榜单分的现实预期

公开榜是测试集的一个子集。6398 行、约 640 个正例时，AUC 的抽样标准误约 0.011；
若公开榜只用一半数据，约 0.016。上一版本地（旧口径）0.7027 对应榜单 0.70722，
这 +0.0045 的差额本身就落在噪声量级内。

所以：本方案相对 B7 的 **+0.008 本地增益是可信、可复现的那部分**，
落到公开榜上会再叠加一个 ±0.01 量级的随机项。合理预期是 **0.711–0.719**，中位数约 0.715。
任何"保证 0.72"的说法都是把榜单噪声当本事。要真正拉开差距，只能继续做 5.1。
