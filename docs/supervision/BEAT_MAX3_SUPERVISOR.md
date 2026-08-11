# Beat-max3 独立严格监督意见

> 角色：只读监督者（不改训练配方）  
> 对象分支：`cursor/beat-max3-6de7`  
> 审计时点：2026-08-11  
> 硬目标：公开榜 **> 0.71222**（`submission_v4_max3.csv`）  
> 已知失败：V4ext 实榜 **0.71123**（丢掉 `ord_noxb_bag` + 诚实 nested 外推）

---

## 1. 裁决摘要

| 项 | 结论 |
|---|---|
| **CRITICAL：fuse 用 mean** | **未发现** — 融合为 `np.maximum.reduce` + 逐臂 `rank01`，即 **`max(rank)`** |
| **CRITICAL：丢掉 `ord_noxb_bag`** | **未发现** — 五份已过门禁 report 均含底座三臂 |
| **G1–G6 执行** | **形式通过**；G4/G6 实现上近乎恒真（见 §5） |
| **交榜放行** | **有条件放行**（仅监督过的 CSV；未回执前不得宣称已超 0.71222） |
| **nested→LB 外推** | **禁止**（见 §3） |

---

## 2. 是否放行交榜（文件与优先级）

### 2.1 放行（须先有 `report_*.json` + `supervisor_*.json` 且 `passed=true`）

| 优先级 | 文件 | nested Δ vs max3 | Spearman | blocks+ | 备注 |
|---:|---|---:|---:|---|---|
| **1（首选）** | `submissions/submission_max3_best.csv` | **+0.00282** | 0.9918 | 4/5 | plus+noxb10+cat_w12_d5 |
| 2 | `submissions/submission_max3_plus_w12.csv` | +0.00238 | 0.9917 | 4/5 | 无第二 ES noxb10，略保守 |
| 3 | `submissions/submission_max3_pro.csv` | +0.00215 | 0.9917 | 4/5 | plus+noxb10 |
| 4 | `submissions/submission_max3_pro_sem.csv` | +0.00250 | 0.9892 | 4/5 | Δ 高但 Spearman 更低，次选 |
| 5 | `submissions/submission_max3_plus.csv` | +0.00146 | 0.9917 | 4/5 | 最小改动 / 兜底 |

**一致性复核**：上述五份 `artifacts/beat_max3/` 与 `submissions/` 字节级 MD5 一致；`report_*` 与 `supervisor_*` 的 delta/nested/spearman/gate 一致；`fusion` 字段均为 `"max(rank)"`；`submission_max3_best.csv` 与冠军 CSV 的 Spearman≈0.99175，与 report 吻合。

### 2.2 不放行

| 文件 | 原因 |
|---|---|
| `submission_max3_plus_str_b7_close_cat_w12_.csv` | 无 `supervisor_*.json`；含 **b7_closest** 筛查产物 |
| `submission_max3_noxb10_plus_v10_cat_w12_.csv` | 无监督落盘；仅 screen |
| `submission_max3_plus_str_noxb10_cat_w12_.csv` | 无监督落盘；仅 screen |
| 任何 `screen_report.json` top 配方直接交榜 | 未经 `supervise.py` 正式门禁写回 |

---

## 3. 对 nested 外推的禁止声明

**禁止**将本地 nested（含相对 max3 的 Δ）加上 **0.0095 / 0.0092** 或任何「诚实 gap」常数后，写成公开榜预估并据此宣称可超 **0.71222**。

依据：

1. V4ext：nested≈0.70381 + gap 外推≈0.713 → **实榜 0.71123**，失败。  
2. `fuse_max3_plus.py` / `supervise.py` 均写明 `lb_claim=null`，且 caution：ES 臂存在时 nested **乐观**。  
3. 历史审计（`docs/SUBMISSION_AUDIT.md`）：max3 含 ES，`artifacts/v2/es_bias.json` 同类乐观约 **+0.00254**；**不得**用诚实尺子解读 ES-nested。

**允许说的话**：本地门禁通过、相对 max3 nested Δ=…、Spearman=…；**实榜未知**。  
**不允许说的话**：预计 LB≈0.71222+Δ、预计 0.714+、已超过冠军。

---

## 4. 协议风险（必须知晓）

### 4.1 ES（early stopping）污染 nested

- 冠军底座含 **`ord_noxb_bag`（ES）**；候选另加 `noxb10` / 训练中的 `ord_noxb_*`（`use_best_model` + `eval_set`）。  
- 协议允许「与冠军同协议的混合 ES」，但 **nested Δ 不可当作诚实增益的无偏估计**。  
- 交榜是赌 raw LB，不是证明诚实抬分。

### 4.2 `b7_closest`

- `screen_report.json` 中 `+plus_strong+b7_closest+cat_w12_d5` 本地 Δ≈**+0.00271**（仅次于 best）。  
- B7 历史公开榜 **≤0.70722**，与 max3 协议不同；高 Spearman 筛查（部分 combo ≈0.995）可能意味着「换血不足」或「与底座过近」。  
- **本轮正式 SHIP 列表未纳入 b7**（正确）。未经独立 supervise + 协议披露，**禁止**因 screen 排名靠前而交 `*_b7_close*`。

### 4.3 `cat_w12_d5` / V4ext 遗产

- best / plus_w12 依赖 `cat_w12_d5`；该臂来自 V4ext 世界（固定树）。  
- V4ext 整包失败 ≠ 单臂必然有害，但 **G5 上 `cat_w12_d5` win_rate≈0.172，仅刚过 0.15 门槛** — 边际臂，下一轮若跌破须剔除。

### 4.4 Block0 系统性回退

五份过门禁配方在 **block0 均劣于底座**（best/pro ≈ −0.0004；plus/sem ≈ −0.0014）。靠其余 4 block 凑够 G3（4/5）。  
这是 **局部不稳信号**：公开折若更像 block0，增益可能蒸发。

### 4.5 监督脚本口径张力

- `supervise.py` note：「Prefer **smallest** delta-passing recipe」  
- `docs/BEAT_MAX3.md` / `LB_BOARD.md`：推荐 **最高** Δ 的 `max3_best`  

监督裁决：**交榜优先级以本意见 §2.1 为准**（首选 best；槽位紧张时 plus_w12/pro 作保守备份）。不得用「最小 Δ」借口跳过已放行首选。

### 4.6 产物写入时机

`fuse_max3_plus.py` **门禁失败仍写 CSV**。因此 `submissions/` 中出现文件 ≠ 放行。交榜前必须以 `supervisor_*.json` 的 `verdict=SHIP_CANDIDATE` 为准。

---

## 5. 门禁 G1–G6 审查

| 门 | 含义 | 本轮结果 | 严格意见 |
|---|---|---|---|
| G1 | Spearman ∈ [0.985, 0.997] | 全过（sem 最低 0.9892） | 有效 |
| G2 | Δnested ≥ 0.001 | 全过 | 有效；**不**等于可交榜宣称超 LB |
| G3 | blocks+ ≥ 4/5 | 全过（均为 4/5） | 有效；掩盖 block0 回退 |
| G4 | 保留 `ord_noxb_bag` | 全过 | **实现脆弱**：`names = BASE + extra`，G4 几乎恒 true |
| G5 | 新臂 win_rate ≥ 0.15 | 全过 | 有效；w12 贴边 |
| G6 | 保留三底座 | 全过 | **同 G4，近乎恒 true** |

**代码核验（`src_beat/fuse_max3_plus.py`）**：

- 融合：`cand_oof/test = np.maximum.reduce([...])`，前置 `rank01` → **确为 max(rank)**。  
- 底座：`BASE_NAMES = ("merger_ord8", "v2_cat_alt8", "ord_noxb_bag")` 强制并入。  
- `np.mean` 仅用于 **win_rate / seed bag 池化**，**未**用于臂间融合。

**门禁漏洞（非本轮 CRITICAL，但记为否决级修复要求）**：G4/G6 应校验「实际参与 `maximum.reduce` 的臂集合」且 ideally 校验 npz 指纹/路径，而非仅检查由代码自己拼出来的 `names` 列表。当前无作弊证据，但门禁不能发现「改 BASE_NAMES 删 noxb」以外的丢臂方式时，监督者必须人工读源码（本次已读）。

---

## 6. 下一轮训练必须满足的否决条件

任一触发即 **REJECT / 不得交榜**：

1. **融合改为 mean / weighted mean / 连续搜权**，或 report 中 `fusion != "max(rank)"`。  
2. **丢掉或替换** `ord_noxb_bag`（含「诚实化」重训后偷换底座同名文件而未复验冠军三臂）。  
3. 底座缺 `merger_ord8` 或 `v2_cat_alt8`。  
4. 用 **nested + 0.0095（或任何固定 gap）** 写进交榜理由或对外宣称 LB。  
5. G1–G3/G5 任一失败，或仅有 CSV、无 `supervisor_*.json` / `passed!=true`。  
6. 以 **b7_closest** 或未过 supervise 的 screen CSV 作为主交榜文件。  
7. 新臂 G5 win_rate **< 0.15** 仍强行并入并宣称 SHIP。  
8. 为抬 nested 而 **扩大 ES 臂数量** 却不在报告中披露 ES 协议，或把 ES-nested 与诚实 nested 混排争第一。  
9. 复现 V4ext 路线：删 noxb + 固定树堆臂 + 外推冲榜。

**正向硬要求（通过才可再申请放行）**：底座三臂冻结进 fuse；`max(rank)`；正式跑 `supervise.py`；`lb_claim` 保持 null 直至公开榜回执。

---

## 7. CRITICAL 扫描结论

| 扫描项 | 结果 |
|---|---|
| fuse 用 mean 融合 | **否** — 无 CRITICAL |
| 丢掉 `ord_noxb_bag` | **否** — 无 CRITICAL |
| 提交与 report 不一致 | **否**（五份正式候选） |
| 未监督 CSV 混入 submissions | **是（WARNING）** — 已列入不放行 |

---

## 8. 签字

独立严格监督者：**有条件放行交榜**。  
首选文件：`submissions/submission_max3_best.csv`。  
**未回执前不得宣称公开榜超过 0.71222。**  
无 CRITICAL（mean / 丢 noxb）；存在 WARNING：未监督 screen CSV、G4/G6 恒真、ES nested 乐观、block0 回退、b7 screen 诱惑。
