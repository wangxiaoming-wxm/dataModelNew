# 方案说明（task-20260808-cursor）

## 0. 一句话

先把赛题数据的生成机制逆向出来，删掉 27 列匿名化噪声、找回被车型量纲掩盖的
`condition` 信号，再用"多扰动重编码 + 多种子 bagging + 多臂融合"的 CatBoost 集成，
在**不在验证折上早停**的诚实协议下把本地 OOF 抬上去。

## 1. 为什么不基于 B7 继续改

B7 的本地权威分 0.702705 是在这样的循环里算出来的（`src/insurance_claim/train_b6.py:321`）：

```python
model.fit(tr, y_tr, eval_set=(va, y_va), use_best_model=True, **fit_kw)
...
oof[va_idx] = model.predict_proba(va)[:, 1]
```

**外层验证折同时被用来选迭代数、又被用来产生 OOF 预测。** 这是一个真实存在的乐观偏差。
我用同一份 gap 特征、同一组参数量化过：

| 协议 | 单模型 OOF | 多分区 bag |
|---|---:|---:|
| 在验证折上早停（B6/B7 协议） | 0.68800 | 0.69111 |
| 固定树数，不看验证折 | 0.68658 | 0.68948 |

偏差约 +0.0015。数值不大，但它让"本地分"失去了作为决策依据的资格——
我这一版所有的对比都在固定树数的协议下做，任何配置都不许看验证折。

顺带说明另一件事：B7 文档里"本地 0.7027 → 公开 0.70722"的 +0.0045，
和"LightGBM 只能到 0.67"这类结论，都是在**同一套被高估的口径**下得出的，
所以我没有沿用它的任何超参结论，而是从数据本身重新开始。

## 2. 数据结构逆向（核心，详见 `docs/DATA_STRUCTURE.md`）

结论摘要：

- `source`（11 类）是隐藏的车型 ID。`V`、`x19`、`code`、`t3`、`cc`、`max_g`、`x0`–`x17`
  这 **27 列**都是它的确定性函数加均匀噪声，组内残差的宽度精确等于 `极差/√12`。
- `livability` ≡ `region`；`x20` ≈ `1.2·condition − 0.53 + U(−1.5, 1.5)`；`x18` 是无条件噪声。
- 把这些残差单独或联合拿去预测标签，全部落在随机置换的噪声带内（联合 LightGBM AUC 0.518，
  AUC 标准误约 0.008）。
- 真正有信号的只有 `days`、`region`、`condition`、`source`、`age_range` 和 8 个二值开关；
  `month` / `version` / `grades` 的诚实折外目标编码 AUC 都在 0.50 附近。
- 对抗验证 train vs test AUC = 0.4977，两边同分布，没有分布偏移可利用，也没有可用的泄漏。

由此得到本方案最重要的一个特征：

```
cond_r = condition / median(condition | source)     # 单列 AUC 0.532 -> 0.567
ratio  = days / cond_r                              # 单列 AUC 0.620
```

`ratio` 比原始最强列 `days`（0.593）还高，是全数据最强的单一排序变量。
`source × condition` 也是所有二阶交叉里增益最大的一项（+0.0292）。

## 3. 模型

### 3.1 特征

`src2/features.py` 的 `cross2` 档：

- 数值：`days`、`condition`、`cond_r`、`ratio` 及其对数/幂次变体、`age_range`、8 个开关、开关和；
- 类别：`region`、`source`、`month`、`version`、`grades`、`age_cat`、8 位开关模式 `bin_pat`、
  `days` 的业务分箱，以及 `days`/`condition`/`cond_r`/`ratio` 的 5/10/20/40 分位分箱；
- 交叉：围绕已验证的交互（`condition×source` 最强）手工构造约 45 个二阶/三阶交叉，
  并对同一个交互给出多档分辨率；
- 频次编码：对关键类别列做 label-free 的计数编码。

### 3.2 一个反直觉的做法：把噪声列加回去

既然 27 列是噪声，直觉是删干净。实测相反：

| 特征集 | 单模型 OOF | bag |
|---|---:|---:|
| 干净交叉集 | 0.67934 | 0.68225 |
| + 主办方的冗余噪声编码 | **0.68921** | **0.69209** |
| + 自造的抖动重编码（`src2/jitter.py`） | **0.69028** | **0.69386** |

原因不是它们含信息，而是它们是同一批真实交互的**不同扰动离散化**
（`x20×source` 就是 `condition×source` 的加噪版本，`t3×days_bin` 就是 `source×days_bin` 的随机细分）。
CatBoost 对每个类别列计算有序目标统计量，多个扰动版本并存 = 树集成对同一统计量做了平均，方差下降。

`src2/jitter.py` 把这个机制变成可控的：用行 `id` 的哈希生成确定性的 `U[0,1)` 扰动流，
造出受控的重编码视图；**每个种子用不同的扰动流**，等于给 bagging 又叠了一层多样性。

这一步不改变"噪声列零信号"的结论，所以在 LightGBM 与 GLM 臂里这些列全部剔除——
那两个模型没有有序目标统计量保护，喂噪声只会过拟合。

### 3.3 四个臂

| 臂 | 模型 | 特征 |
|---|---|---|
| `cat_d5` | CatBoost depth 5 / 1000 树 / lr 0.03 | cross2 + 噪声视图 + 抖动视图 |
| `cat_d6` | CatBoost depth 6 / 700 树 / `bagging_temperature=1` | 同上 |
| `lgb_te` | LightGBM + 折内嵌套目标编码 | 仅有信号的列 |
| `glm` | 样条 + 目标编码 logit 的 L2 逻辑回归 | 仅有信号的列 |

目标编码由**拟合行内部的一层 K 折**产生，任何一行都不会看到自己的标签。

### 3.4 协议

- 重复分层 5 折，所有臂共用完全相同的分区；
- 固定树数，**不在外层验证折上做任何早停或选择**；
- 全部特征工程 label-free（分位切点、车型量纲、频次、抖动流），因此在 train+test 上拟合一次是安全的；
- 臂内 OOF = 各种子秩平均；臂内 test = 每个 (种子, 折) 模型的秩平均；
- 融合规则从**预先登记**的一小组权重里用**嵌套选择**挑：OOF 分成 5 个外层块，
  在其中 4 块上选规则、在第 5 块上应用，最终 AUC 在这样拼出来的预测上计算，不含选择乐观。

## 4. 复现

```bash
pip install -r requirements.txt

# 逐段跑 OOF（每段 4 个种子，抖动流互不重叠）
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20260 20261 20262 20263 --stream-base 0  --out artifacts/v2a
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20264 20265 20266 20267 --stream-base 10 --out artifacts/v2b
PYTHONPATH=src2 python3 src2/run_oof.py --seeds 20268 20269 20270 20271 --stream-base 20 --out artifacts/v2c

# 合并成 12 种子的 bag，再融合并写提交
PYTHONPATH=src2 python3 src2/merge_runs.py --inputs artifacts/v2a artifacts/v2b artifacts/v2c --out artifacts/v2
PYTHONPATH=src2 python3 src2/fuse.py --dir artifacts/v2 --submission submissions/submission_v2.csv
```

## 5. 结果

见 `docs/RESULTS.md`。
