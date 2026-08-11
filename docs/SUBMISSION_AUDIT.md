# 独立提交审核报告（对抗式）

审核日期：2026-08-11  
分支：`cursor/honest-push-v4ext-6de7`  
立场：不继承交付方叙事；已测 LB 与仅本地 nested 分列；含 ES 的 nested 不得与诚实 nested 混排。

---

## 0. 硬锚（用户核实公开榜，2026-08-10）

| 文件 | LB | 本仓库/zip 状态 |
|---|---:|---|
| `submission_v4_max3.csv` | **0.71222 ★** | `/tmp/audit_all/...` 可复现；含 1×ES 臂 |
| `submission_v5_honest.csv` | 0.71207 | zip 可复现；纯诚实 |
| `submission_v3_max3.csv` | 0.71184 | zip 存在 |
| `submission_v4_honest.csv` | 0.71104 | zip 可复现；纯诚实 |
| `submission_v3.csv` | 0.71064 | `submissions/` |
| `submission_v5.csv` | 0.71035 | `/tmp/audit_all/v5/` |
| `submission_v2.csv` | 0.70878 | `submissions/` |
| B7 closest 等 | ≤0.70722 | `submissions/submission_b7_closest_honest.csv` |

**未测（本轮需排序的本地候选）**：`submission_v4ext.csv`、`submission_v4.csv`、以及 zip/分支中的 `v4max3pro` / `v4max3pronew`。

---

## A) 过拟合 / 协议风险表

| 候选 | 协议 | 本地尺子 | 已测 LB | 风险 | 证据 |
|---|---|---:|---:|---|---|
| **v4ext** `submissions/submission_v4ext.csv` | 诚实（固定树，无 ES） | nested **0.7038146**（现场 fuse4/audit 复算一致） | **未测** | **低–中** | `honesty_passed=true`；opus 三臂 OOF/test 与 v5_honest 字节级一致；`run_world.py` 写明 fixed iterations / 无 eval_set；选择乐观 full−nested=**0.000481**；29 规则扫描但 nested 只在 `ma_w12`(52)↔`mag_w12`(48) 间抖；vs max3 Spearman **0.99407**、vs v5_honest **0.99443**（排序有别但非大换血） |
| **V4 main** `submissions/submission_v4.csv` | 诚实 | nested **0.7030285**（`artifacts/v4/fusion_report_v4.json` / audit_v4） | **未测** | **低** | 规则 `views_max_10_20_r16_r16b` 可逐元素复现 CSV；乐观 0.00025；被 v4ext 本地严格支配 |
| **v5_honest** | 诚实 | nested **0.70253**（rank_all / README） | **0.71207** | **低（已校准）** | 三臂固定树；实测 gap=+0.00954 |
| **v4_honest** | 诚实 | nested **0.699926** | **0.71104** | **低（已校准）** | gap=+0.01111；更老、更弱 |
| **v4_max3** | **混合 ES**（1 臂 `ord_noxb_bag`） | nested **0.703073**（**乐观上界**） | **0.71222 ★** | **中（协议）** | README 自承 ES；`artifacts/v2/es_bias.json` 同类协议 AUC-ES 乐观 **+0.00254**；**不可与诚实 nested 比高低** |
| **v3_max3** | 混合 ES（历史） | （未在本分支独立复算） | **0.71184** | **中** | 已测；再交无信息增益 |
| **v4max3pro** | **混合 ES**（含 `ord_noxb_bag` + `noxb10` 明确 `use_best_model`/`eval_set`） | nested **0.705220**（ES 污染） | **未测** | **高** | `recipe_report` 自标 `es`/`plus10`；用 max3 gap 外推 LB≈0.7144 属**选择乐观+协议污染**；Spearman vs max3 **0.9917** |
| **v4max3pronew** | **混合 ES**（pro + `semantic_rmse` protocol=`es5_rmse_bag10`） | nested **0.705570**（ES 污染） | **未测** | **高** | 扫描多 combo 后取 nested 最高；notes 写明 ES；外推 LB≈0.7147 **未校准**；Spearman vs max3 **0.9892** |
| **v4_es / max4** 等 | ES | — | 未作为主锚 | **高** | 与 max3 Spearman≥0.999；无额外已测优势 |
| **artifacts/v4_ext/submission_v4ext.csv** | （旧快照） | — | — | **勿交** | 与正式 `submissions/submission_v4ext.csv` **不等**：Spearman 0.9970，max\|Δ\|=0.23 |

### 选择乐观 / 扫描宽度（V4ext）

| 量 | 数值 | 来源 |
|---|---:|---|
| 可用规则数 | 29 | 现场 `fuse4.py` |
| full 最优 | 0.7042956 (`views_max_v4_ma_w12`) | fusion_report |
| nested mean±sd | 0.7038146 ± 0.0001445 | 现场复算 |
| full−nested | **0.000481** | 复算 |
| pick_counts | ma_w12:52 / mag_w12:48 | 复算 |
| 强家族臂间秩相关 | 0.973 | `artifacts/audit_v4/evidence_v4.json` |

结论：存在规则扫描，但 nested 已吃掉大部分选择乐观；**未见「本地猛涨、test 排序几乎不变」到需要直接否决的程度**（相对 max3 的 rank MAD≈122，高于 v5_honest 对 max3 的 65）。

### ES bias 锚

`artifacts/v2/es_bias.json`（同折对照）：early_stop AUC **0.6871** vs fixed_500 **0.68456** → 乐观 **+0.00254**。  
因此 **pro/pronew/max3 的 nested 抬升不可直接当成诚实泛化增益**。

---

## B) 推荐提交顺序（最多 5 槽）

仅排**尚未公开测过**、且仍有信息量的文件：

| 序 | 文件 | 为何排这里 |
|---|---|---|
| **1** | `submissions/submission_v4ext.csv` | 诚实协议下本地最高 nested（**0.70381**）；现场 fuse4 重建 CSV **逐元素一致**；`honesty_passed`；相对已测诚实冠军 v5_honest（0.70253→0.71207）有 +0.00128 nested 空间，是校准 gap 的唯一正确下一步 |
| **2** | （可选，投机）`/tmp/audit_all/v4max3pro/.../submission_v4max3pronew.csv` | 仅当槽还多、且接受 **ES 协议** 赌 raw LB；本地 ES-nested 最高，但**不能**用诚实尺子解读 |
| **3** | （可选，投机）`.../submission_v4max3pro.csv` | 被 pronew 本地支配；信息量更低 |
| **4** | `submissions/submission_v4.csv` | 诚实但被 v4ext 支配；仅在 v4ext 异常（文件/榜错）时作对照，**默认不交** |
| **5** | — | 不建议填满；勿为凑槽交已测文件 |

**一句话**：下一步只交 `submission_v4ext.csv`；pro/pronew 仅作协议外投机，不能排在诚实候选之前。

---

## C) 明确不建议交的名单

| 文件 | 原因 |
|---|---|
| `submission_v4_max3.csv` 及等价副本 | **已测 0.71222**；再交零信息 |
| `submission_v5_honest.csv` | **已测 0.71207** |
| `submission_v3_max3.csv` / `v4_honest` / `v3` / `v5` / `v2` / B7 | 均已测且更低 |
| `artifacts/v4_ext/submission_v4ext.csv` | **过期副本**，≠正式 submissions |
| `submission_v4ext_w12.csv` | 与 `submission_v4ext.csv` **字节相同**（sha256 `a755291d30e091e2…`），重复交 |
| 以「冲 0.725」为名的任何 ES/pro 配方 | Bayes/缺口论证见 STATUS；换尺子 ≠ 诚实达标 |

---

## D) V4ext 预估 LB 是否仍成立

### 现场核对（非交付方口述）

```
fuse4 --dir artifacts/v4  → nested_oof_mean = 0.7038146386493205
audit_v4                  → honesty_passed=true, 同 nested
rebuild CSV vs submissions/submission_v4ext.csv → max abs diff = 0
```

### 用已校准诚实 gap 外推（禁止用 max3 的 ES-nested gap 当主尺子）

| 锚 | nested | LB | gap | → v4ext 外推 LB |
|---|---:|---:|---:|---:|
| v5_honest | 0.70253 | 0.71207 | **+0.00954** | **0.71335** |
| V3 | 0.701245 | 0.71064 | +0.00940 | 0.71321 |
| repo V5 | 0.700757 | 0.71035 | +0.00959 | 0.71341 |
| V2 | 0.69856* | 0.70878 | +0.01022 | 0.71403 |
| v4_honest | 0.699926 | 0.71104 | +0.01111 | 0.71493（偏松，历史已证明易高估） |
| v4_max3（ES，**仅对照**） | 0.703073 | 0.71222 | +0.00915 | 0.71296 |

\*V2 nested 取 PLAN_AUDIT/历史诚实锚；rank_all 5-block 尺为 0.698784，不混用进主外推。

### 裁决

- 交付方写的 **0.7130–0.7133（gap 0.0092–0.0095）与最贴的诚实锚（v5_honest / V3 / V5）一致，区间仍成立。**
- **不是承诺。** 风险下行：若高 Spearman（≈0.994）导致榜上增益弱于 nested 增益，可能落在 **0.7120–0.7125**（贴近现冠军）。
- 历史教训：v5_honest README 曾用 v4_honest gap 外推 ~0.7136，**实测 0.71207**（高估 ~0.0015）。故 **弃用 +0.011 类宽松 gap**；主预估带宽保持 **≈0.7130–0.7134**。
- 相对现冠军 0.71222：期望约 **+0.0008～+0.0011**（与 STATUS 同量级），**非保证**。

---

## 附录：关键复算命令与 Spearman

```bash
python3 src4/fuse4.py --dir artifacts/v4 \
  --submission /tmp/audit_v4ext_rebuild.csv \
  --report /tmp/fuse4_v4ext_verify.json
python3 src4/audit_v4.py --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --out /tmp/audit_v4ext_live.json
```

| 对比 | Spearman |
|---|---:|
| v4ext vs v4_max3 | 0.994070 |
| v4ext vs v5_honest | 0.994431 |
| v4ext vs V4 | 0.999347 |
| v4max3pro vs v4_max3 | 0.991735 |
| v4max3pronew vs v4_max3 | 0.989202 |
| v4_es vs v4_max3 | 0.999784 |

---

## 总结

1. **诚实未测第一优先：`submissions/submission_v4ext.csv`。**  
2. ES 族（max3 已测；pro/pronew 未测）nested 更高但**尺子污染**，不得压过诚实排序。  
3. V4ext 预估 LB **0.7130–0.7133 仍与已校准诚实 gap 相容**；交榜校准前不得写成承诺。
