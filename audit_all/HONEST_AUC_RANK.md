# 全分支 / zip 方案诚实 AUC 审核与提交顺序

审核日期：2026-08-10  
尺子（本地复算）：**5-block nested AUC**（`np.array_split` 连续块，块内重排后再算 AUC）；融合统一为 `max(rank(·))`（与各包文档一致时再 `clip[0.001,0.999]`）。  
原则：**只写能核对的事实**；公开榜只收「文档写明已实测」的数字；含早停（ES）的 nested **不得**当作无偏 AUC 与诚实方案直接比高低。

复算脚本产出：`audit_all/rank_all.json`（完整）、`audit_all/rank_all_slim.json`（摘要）。

---

## 0. 覆盖范围

| 来源 | 内容 | 有可提交 CSV？ |
|---|---|---|
| `origin/20260810-5fangan` → `715.zip` | 仅 `feat_semantic.py` / `explore_d_online.py` 探索脚本 | **无**（脚本会写提交，但 zip 内无产物、无 OOF） |
| `origin/20260808-cursor-opus-grok-glm` → `20260810-cursor-opus5.zip` | `v4_honest` / `v4_max3` / `v5_honest` 三套完整包 | 有 |
| `origin/zcode-v4-max3` → `20260808-zcode-cursor.zip` | 多阶段提交 + 臂 npz | 有 |
| `main` / `cursor/honest-auc-v4-145a` | V2/V3/V4 诚实管道 | 有 |
| `cursor/honest-auc-push-f126` | V5 诚实（V2 同尺子） | 有 |
| `cursor/v4max3pro-f126` / `v4max3pronew-f126` | max3 + ES 增臂 / + semantic_rmse | 有 |
| `cursor/audit-5-submissions-3057` / `audit-v4max3pro-83bb` | 既有独立审计（本报告交叉引用其可复核结论） | — |
| `task-20260809-cursor-gpt56` | v2/B7 冻结混合与 10-fold 试验 CSV | 有（**无独立公开榜实测**） |
| 其余 task/copy 分支 | V2/B7 谱系，不新增更强已验证提交 | — |

数据：`train.csv` / `test.csv` SHA256 与仓库 `data/SHA256SUMS.txt` 一致（本环境 train=`494a61073a0438f6…`）。

---

## 1. 已核实的公开榜（唯一硬锚）

| 提交 | 公开榜 | 证据 |
|---|---:|---|
| `submission_v4_max3.csv` | **0.71222** | opus / zcode / `v4max3pro` 三份 **SHA256 完全相同**（`01afc280d1a478a5…`）；zcode REPORT 与 opus README 均记为实测 |
| `submission_v3_max3.csv`（zcode） | **0.71184** | zcode `REPORT.md` 写明已验证 |
| `submission_v2.csv` | **0.70878** | 多分支 README / RESULTS 一致记载 |
| B7 closest | **0.70722** | 同上 |

**不得当作事实的声明：**

- opus `v5_honest` README：「v4_honest 实测 LB 0.71104」——既有 `audit5/FINDINGS.md` 已标明**未获独立确认**；本审核仍无新的榜单回执。
- 由此外推的「v5 预期 LB ≈0.7136」、pro「外推 ≈0.71437」——**不是测量值**。

---

## 2. 本审核独立复算的 nested（同一尺子）

### 2.1 诚实协议（固定树数、无 `use_best_model`）

| 候选 | nested_5block | 来源 | CSV 复现 |
|---|---:|---|---|
| **main `submission_v4.csv`** | **0.70303**（官方 `fusion_report_v4` / `audit_v4`；`honesty_passed=true`） | main；`fuse4` 重建与提交 **字节一致** | 是 |
| opus/zcode `v5_honest` | **0.70253** | zip；两份 CSV **SHA 相同** | max(rank) 可 bit-exact |
| main V3 | 官方嵌套均值 **0.70124**（本审核对三臂 views_max 重建 0.70111，差在嵌套选规则） | main | — |
| repo V5 | 监督者 20-seed 均值 **0.70076**；seed99 嵌套选择 0.70059（`fusion_report.json`） | `honest-auc-push-f126`；views_max(d5,d6,alt,gap) 重建 0.70083 | — |
| opus/zcode `v4_honest` | **0.69993** | zip；两份 SHA 相同 | 是 |
| main V2 | 管道嵌套 **0.69856**；监督均值 0.69824 | main；公开榜 0.70878 | — |

打乱标签：上述可重建融合 OOF 的 perm AUC 均落在 ~0.50 附近（见 `rank_all.json`）。

### 2.2 含早停臂（nested 为乐观上界，不可与 2.1 直接比）

| 候选 | nested_5block | ES 臂数（代码核实） | 与 max3 提交 Spearman |
|---|---:|---:|---:|
| `v4max3pronew` | **0.70557** | ≥3（继承 pro）+ semantic_rmse | 0.98920 |
| `v4max3pro` | **0.70522** | 3/5（`ord_noxb_bag`/`noxb10`/`plus_strong` 均 ES） | 0.99174 |
| zcode `v4_max4` | **0.70321** | 2（`ord_noxb_bag`+`ordered_bag` 均 `use_best_model=True`） | **0.99978** |
| opus/zcode/repo `v4_max3` | **0.70307** | 1/3 | 1.0（自身） |

同折同种子对照（仓库 `artifacts/v2/es_bias.json`）：AUC 早停相对固定树 **+0.00254** OOF 乐观。  
`ord_noxb_bag.py` / `ordered_bag.py` 现场可见 `eval_set` + `use_best_model=True`。

---

## 3. 关键去重事实

1. **`v4_max3` 三源同文件**：opus zip = zcode zip = `cursor/v4max3pro-f126` 的 `submission_v4_max3.csv`（同 SHA）。公开榜 0.71222 对三者同时成立。  
2. **`v4_honest` / `v5_honest`**：opus 与 zcode 对应 CSV 亦同 SHA。  
3. **`v4_max4` 几乎不改排序**：相对 max3 Spearman 0.99978 → 在已有 0.71222 时，再交 max4 **没有可辩护的增量证据**。  
4. **`v4max3pro` 既有独立审计裁决为 `PROTOCOL_RISK`**（`audits/AUDIT_V4MAX3PRO.md`）：可复现、无作弊；但 nested +0.00215 含 ES/选择乐观；`noxb10` 贡献的本地增益几乎不改 test 排序（Spearman 0.99973）；审计意见为**不建议交**。本审核复算 nested 与之一致（0.70522）。  
5. **`v4max3pronew`**：在 pro 上再加 `semantic_rmse`，nested 0.70557；分支 README 写明 logloss 多样性进 max **降低** nested，未进最终配方。仍属 ES 混合协议，无公开榜。  
6. **`715.zip`**：不是方案交付物，无法进入 AUC 排序。  
7. **gpt56 CSV**：是对已有 v2/B7（及 10-fold 试验）的冻结混合；相对 max3 Spearman ≈0.990–0.994；**无公开榜实测、无高于 max3 的诚实 nested 证据**。  
8. **V3 本地涨、榜上输给 V2**：`docs/V5.md` 明确记载——说明**即使诚实 nested，也不能外推必涨榜**。

---

## 4. 诚实提交顺序（不过拟合）

排序键（硬约束）：

1. **已实测公开榜优先**（事实 > 本地分）；  
2. 未上榜者只按 **诚实 nested** 排序；  
3. **禁止**用含 ES 的 nested 插队到诚实方案之上；  
4. 与已验证冠军 Spearman≥0.9997 的近副本不占用名额。

### 推荐提交队列

| 序 | 文件 | 理由（仅事实） |
|---:|---|---|
| **1** | `submission_v4_max3.csv`（opus/zcode/repo 任一同 SHA） | **已验证公开榜最高 0.71222**；本地 nested 0.70307 含 1 个 ES 臂，但榜分是硬事实 |
| **2** | `submissions/submission_v4.csv`（main / V4） | 未上榜候选中 **诚实 nested 最高 0.70303**；`audit_v4`：`honesty_passed=true`，无早停；`fuse4` 与 CSV 一致 |
| **3** | opus/zcode `submission_v5_honest.csv` | 诚实 nested **0.70253**（本审核复算）；三臂均固定树；次于 V4、高于 V3/repo-V5 |
| **4** | （可选，低优先）`submission_v3.csv`（main） | 诚实嵌套均值 0.70124；但已有「本地高于 V2、榜上输给 V2」的先例，不宜高预期 |
| **5** | （可选）repo `submission_v5.csv` | 诚实监督均值 0.70076；低于 opus `v5_honest` |

### 明确后置 / 不建议占用名额

| 文件 | 原因 |
|---|---|
| `submission_v4_max4.csv` | 相对 max3 Spearman 0.99978，增量不可辩护 |
| `submission_v4max3pro.csv` / `…pronew.csv` | nested 更高但属 ES+选择乐观；独立审计 `PROTOCOL_RISK`，建议不交 |
| `submission_v4_honest.csv` | 诚实但 nested 0.69993，弱于 V4 / v5_honest；其「LB 0.71104」未核实 |
| `submission_v2.csv` / B7 / `submission_v3_max3.csv` | 公开榜均 **低于** 已验证的 0.71222 |
| gpt56 五个 CSV | 无更强诚实证据、无更高实测榜 |
| `715.zip` | 无提交文件 |

### 若只能再交 **一个** 未测文件

交 **`submissions/submission_v4.csv`（main V4）**：在全部「无公开榜 + 诚实协议」候选里 nested 最高，且监督门通过。  
不交 pro/pronew：其更高 nested **不是**同口径无偏 AUC。

---

## 5. 分层总表（去重后）

| 层 | 排序 | AUC 口径 | 值 |
|---|---|---|---:|
| A 实测榜 | 1 max3 → 2 v3_max3 → 3 V2 → 4 B7 | 公开榜 | 0.71222 / 0.71184 / 0.70878 / 0.70722 |
| B 诚实本地 | 1 main V4 → 2 opus v5_honest → 3 main V3 → 4 repo V5 → 5 v4_honest | nested（无 ES） | 0.70303 / 0.70253 / 0.70124 / 0.70076 / 0.69993 |
| C ES 本地（参考，不参与诚实排序） | pronew → pro → max4 → max3 | nested（含 ES） | 0.70557 / 0.70522 / 0.70321 / 0.70307 |

**最终提交顺序（合并 A 优先 + B 补位）：**  
**max3 → main V4 → opus v5_honest**；（其后仅在名额富余时考虑 main V3 / repo V5；pro/max4/honest-baseline/gpt56/715 不进入争榜主队列。）
