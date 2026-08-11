# SUPER714 独立审核报告

审核对象：GitHub PR #12（`cursor/super714-edb2` → `real714`）

审核范围：README、运行入口、依赖、`src_super/*`、主提交、冻结产物、全部方案台账、ZIP/参考包、训练/验证测试。

## 1. 总评

**REQUEST_CHANGES**

这不是一个“超越全部方案”的交付，而是对 `real714.zip` 中已知冠军
`best_v1`（线上 AUC **0.71453**）的干净复现，并附带一个尚未完成完整验收的
TE 候选臂。它确实超过了台账中的 `v4_max3`（0.71222）等旧方案，但与
`real714` 冠军本身完全相同，不能宣称相对全部方案实现了超越。

## 2. 用户要求逐条判定

### 1）结合所有方案（含 ZIP）给出超越全部方案的最强方案

**FAIL**

证据：

- `docs/SCHEME_INVENTORY.md` 把 `real714/v1_best714` 记录为线上 **0.71453**
  的当前冠军，并明确列出 `/workspace/real714.zip`。
- `/workspace/real714.zip` SHA-256 为
  `ed4cf82bf2ee8dd8113877c23986b95a374b1faae92c4746c7bd64ee9d4d1b9f`；
  其中的 `best_oof.npy`、`best_test.npy` 和提交与本 PR 产物一致。
- 实际比较结果：本 PR 的 `best_v1_oof.npy`/`best_v1_test.npy` 与
  `real714_pkg/v1_best714/artifacts/best_{oof,test}.npy` 逐数组相等；
  主提交哈希也均为
  `89250a77fa3196eafe8898353b925d78fb23dd10e1a310e1fa04e0067248f88c`。
- `bash run_super714.sh --verify` 复算的 max2 OOF AUC 为 **0.70128**，
  对应线上锚点仍是 **0.71453**，不是高于 0.71453 的新榜单结果。

因此：

- 相对 `v4_max3`、`v5_honest`、`v3_max3`、`v4ext`、`v4_honest`：
  **PASS（本地台账中的已知线上成绩均低于 0.71453）**。
- 相对 `real714.zip` 的 0.71453：
  **FAIL（逐字节/逐数组复现冠军，不是超越冠军）**。
- `docs/ML_DONE.md` 将其称为“当前最强可复现方案”是可以成立的；
  将其理解为“超越全部方案”则不成立。

### 2）代码和文档整理干净、提交完毕

**FAIL**

证据：

- PR #12 的实现、文档、数据、产物均已形成 5 个提交，提交链完整。
- 但当前 `/workspace` 在审核前已有未提交的 `.gitignore` 修改：
  `git status` 显示 `M .gitignore`。这使当前工作树并非干净状态。
- 该修改不属于 `cursor/super714-edb2` 的 HEAD 提交，应在最终交付前明确提交或还原；
  不能把工作树状态称为“整理干净、提交完毕”。

### 3）GitHub 代码可复现（秒级 verify + 训练入口）

**PASS（但验证器需要补强）**

证据：

- 实际执行 `bash run_super714.sh --verify` 成功，耗时约 2 秒：
  `PASS: SUPER714 预计算锚点与主提交验收通过`。
- 验证到 train 14930 行、test 6398 行、submission 6398 行；
  OOF 为 main `0.69992`、alt `0.69770`、fuse `0.70128`。
- `run_super714.sh` 提供 `--verify`、`--smoke`、完整训练和
  `--baseline-only`；完整训练入口存在，数据目录可由 `DATA_DIR` 或
  `--data-dir` 覆盖。
- `python3 -m unittest discover -s tests -v` 的 3 个 TE 测试全部通过。
  `pytest -q` 不可执行是因为 requirements 没有安装 pytest；这不影响已有
  unittest 测试结果，但应在文档中说明测试命令。

### 4）两个 agent 协同（数据挖掘 + ML）且有独立审核

**PASS（证据强度为文档级）**

证据：

- `docs/SCHEME_INVENTORY.md`、`docs/ARM_SELECTION.md` 和
  `docs/HANDOFF_TO_ML.md` 明确记录了数据挖掘侧的方案筛选、台账和向 ML
  的交接边界。
- `docs/ML_DONE.md` 记录了 ML 侧的实现、训练入口、冻结产物和 TE 结果。
- 当前审核由独立审核角色对目标分支、参考 ZIP、产物哈希、运行结果和测试
  重新检查；审核结论不依赖实现者自报的 smoke 日志。
- 仓库没有可验证 agent 会话日志，因此“两个 agent”主要是提交/文档证据，
  不是可机器验证的运行时审计链；这不改变本条在现有材料下的 PASS。

### 5）约束：不过拟合/不作弊/CatBoost/无外部数据/不碰测试标签/无伪标签/折内 FE

**FAIL（严格按用户给出的“折内 FE”约束）**

逐项判定：

- 不过拟合：**条件 PASS**。没有发现监督 stacking、榜单闭环或为过门槛
  扫描大量候选；但当前 TE 只完成 smoke，不能证明完整 8 seed 结果稳健。
- 不作弊：**PASS**。未发现外部请求、公开榜探测或伪造标签逻辑。
- CatBoost：**PASS**。训练臂使用 `CatBoostRegressor`。
- 无外部数据：**PASS**。训练入口只读取给定 `train.csv`、`test.csv`、
  `submit_sample.csv`；`real714_pkg` 只是仓库内参考包。
- 不碰测试标签：**PASS**。验证集/test 的 TE 只接收外层训练折标签；
  没有 `test` 标签输入。
- 无伪标签：**PASS**。代码和文档均未生成伪标签。
- 折内 FE：**FAIL（严格口径）**。`train_super714.py` 在
  `main()` 中用 `raw_all=train+test` 拟合 `edges_main`，并在
  `run_arm()` 中对合并后的 train+test 构造主/次臂特征；这与
  `real714` 的原始实现一致且属于无标签的 transductive FE，但不是严格的
  外层 fold-local FE。另有 `features_te.py:158` 先用整个外层 fit fold
  拟合 days 分箱，再做 inner OOF，分箱边界也没有在 inner fold 内重新拟合。

说明：上述问题不是“测试标签泄漏”。TE 的目标统计本身是折内的，训练行使用
inner OOF 编码；问题是它不满足用户明确写出的最严格 fold-local FE 定义。
若项目协议允许无标签 train+test 变换，应在约束文档中明确写成
“label-free transductive FE 允许”，否则应改为每个外层/内层 fit fold 单独拟合。

## 3. 关键缺陷（按严重度）

### HIGH：用 smoke 结果替代完整预注册门槛，结论不成立

位置：`docs/TE_GATE_RESULT.md:3-11`、`artifacts/super714/metrics_smoke.json`、
`src_super/train_super714.py:419-425,504-565`。

预注册协议要求在同一套 14930 行 pooled OOF、8 seeds × 5 folds × 3 bags
上判断：

1. `TE AUC > 0.697`；
2. `Spearman(TE, main) < 0.90`；
3. `max3 - max2 > 0.001`。

实际只执行了 2 folds × 1 seed × 1 bag 的 smoke，得到 TE AUC `0.67661`、
Spearman `0.88223`、增益 `-0.00371`。代码通过
`accepted = (not args.smoke) and all(gates.values())` 保证 smoke 不覆盖主提交，
这个安全行为是正确的；但文档把 smoke 结果写成“拒绝 TE 第三臂”，并没有运行
完整候选臂，不能证明完整预注册门槛失败。

### HIGH：秒级验证器没有验证主提交等于冻结 best_v1 test fuse

位置：`src_super/verify_super714.py:83-90`、`artifacts/super714/manifest.json:23-25`。

验证器只检查主提交的列、ID、行数、有限性和 `[0.001, 0.999]` 范围；
没有比较：

```python
submission["label"] == np.clip(test_pred["fuse"], 0.001, 0.999)
```

也没有使用 manifest 中已有的 `submission_super714.csv` SHA-256。
因此任意一份 ID 正确、数值范围正确但预测完全不同的 CSV 都可能通过
`--verify`。当前文件经独立计算与冻结 fuse 的最大绝对差为
`1.11e-16`（CSV 浮点序列化造成的 ULP 误差），实际内容是正确的；
但验证器没有把这个事实锁住。

### MEDIUM：严格 fold-local FE 与当前实现/文档口径不一致

位置：`src_super/train_super714.py:435-436`、`src_super/train_super714.py:215-219`、
`src_super/features_te.py:158-159`。

主/次臂的 quantile、source condition median、频次特征由合并后的
train+test 一次拟合；TE 的 days bin 边界则由整个 outer fit fold 拟合后
复用于 inner OOF。它们不读取测试标签，所以不是目标泄漏，但若“折内 FE”
是硬约束，就不应采用该实现。台账应明确区分“无标签 transductive FE”
和“严格 fold-local FE”，或改造代码。

### MEDIUM：主张与交付状态的措辞需要收紧

`docs/ML_DONE.md` 和 PR 标题整体容易让读者把“最强可复现”
理解为“已超越全部方案”。应明确写成：

> `best_v1` 冠军复现（0.71453）；相对 real714 未超越；TE 完整门槛尚未完成。

## 4. 最小修复清单（实现者可直接改）

1. **先修正结论措辞**：README、PR 描述和 ML_DONE 明确标注
   “复现 0.71453，不是超越 0.71453”；保留对旧 v4_max3 等方案的超越说明。
2. **运行完整 TE 候选**：执行默认的 8 seeds × 5 folds × 3 bags，保存
   `metrics.json`，仅以完整 pooled OOF 三项门槛决定是否准入。若未运行，
   `TE_GATE_RESULT.md` 只能写“smoke 未准入、完整门槛未评估”，不能写成
   “TE 已被拒绝”。
3. **补强 `verify_super714.py`**：读取 manifest 的 submission SHA，并同时断言
   提交 label 与 `best_v1_test["fuse"]` 的 `[0.001,0.999]` clip 一致
   （允许明确的 CSV 浮点容差）；继续保留 ID/行数/有限值检查。
4. **落实折内 FE**：若用户约束不允许 transductive FE，则把 main/alt 的
   edges、频次和 TE days edges 都改为对应 fit fold 拟合；并补一个测试证明
   改变 outer/inner holdout 的分布不会影响 fit 行的变换参数。
5. **清理工作树**：处理当前未提交的 `.gitignore` 修改，确保最终交付状态
   `git status --short` 为空，并把必要改动提交到 PR 分支。

## 5. 复核命令与结果

```text
bash run_super714.sh --verify
PASS
train=14930, test=6398, submission=6398
OOF AUC: main=0.69992, alt=0.69770, fuse=0.70128

python3 -m unittest discover -s tests -v
Ran 3 tests ... OK
```

`best_v1_oof.npy` SHA-256：
`ef23c61013f9ecf469174c55849983677de2b669cce6c052f999808545b7600d`

`best_v1_test.npy` SHA-256：
`aaa43ca48b9d297c35367c873f6001c3607f5cff4a9f96a6ac72e284a57942dd`

## 6. 复审（2026-08-11）

复审对象：`cursor/super714-edb2`（PR #12）

已重新执行：

```text
bash run_super714.sh --verify
PASS: SUPER714 预计算锚点与主提交验收通过
OOF AUC: main=0.69992, alt=0.69770, fuse=0.70128
rows: train=14930, test=6398, submission=6398
submission_vs_fuse_max_abs: 1.110e-16

python3 -m unittest discover -s tests -v
Ran 3 tests ... OK

git status --short
（空）
```

逐项复核结果：

- README、`docs/PROTOCOL.md`、`docs/ML_DONE.md` 已明确主交付是复现
  `0.71453`，不是超越；相对旧方案的比较保持清楚。
- `docs/PROTOCOL.md` 已披露 best_v1 使用 label-free transductive FE，
  并区分 TE 候选臂的折内目标统计。
- `src_super/verify_super714.py` 现在强制校验提交 SHA-256，并校验提交
  label 与冻结 `best_v1_test["fuse"]` 的 clip 结果一致；本次运行通过。
- `docs/TE_GATE_RESULT.md` 已将 smoke 标记为非最终判定，并明确完整
  `8×5×3` 门槛须等待 `artifacts/super714/metrics.json`。
- 当前工作树干净，未发现上次报告中的 HIGH 缺陷仍未修复。

基线交付可 APPROVE。完整 TE 尚未产出 `metrics.json`，不影响 best_v1
基线验收，但不得据此宣称 TE 已接受或主交付已超越 `0.71453`。

PENDING: full TE metrics.json

AUDIT_STATUS: APPROVE
