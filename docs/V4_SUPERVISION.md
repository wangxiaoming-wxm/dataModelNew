# V4 独立监督者定稿

Branch: `cursor/honest-auc-v4-145a`（已整理交付）

范围：对 `artifacts/v4` 与 `submissions/submission_v4.csv` 做对抗式审核。  
不修改 V3 文件；不使用测试集标签；不把榜单分当反馈。

## 命令

```bash
bash run_v4.sh
# 或
PYTHONPATH=src2:src3:src4 python3 src4/fuse4.py \
  --dir artifacts/v4 --submission submissions/submission_v4.csv
PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 --submission submissions/submission_v4.csv \
  --target 0.707 --out artifacts/audit_v4/audit.json
```

`audit_v4.py` 通过 `from fuse4 import RULES` 与融合器共用规则表，避免「监督口径 ≠ 提交口径」。

## 门禁表

| 门 | 结果 |
|---|---|
| 数据 SHA256 | PASS |
| 协议扫描（`src4`：无早停 / 无 eval_set / 无 test 标签） | PASS |
| 臂存在且 sanity（AUC∈(0.55,0.90)） | PASS |
| 嵌套选择可运行 | PASS |
| 打乱标签无信号 | PASS |
| 提交格式 | PASS |
| 目标 ≥ 0.707 | **FAIL** |

`honesty_passed = true`，`target_reached = false`。

## 背书主报

```text
nested_oof_mean = 0.7030285030340446
nested_oof_sd   = 0.0001551649037618552
submitted_rule  = views_max_10_20_r16_r16b
best_full_oof   = 0.7033678813195666
honesty_passed  = true
target_reached  = false
gap_to_0.707    ≈ 0.00397
vs_V3_nested    ≈ +0.00179
```

与 `artifacts/v4/fusion_report_v4.json` 一致。

## 意见

1. 上述数字可作为本停止点的**诚实本地 AUC** 背书。  
2. **不得**宣称达到 0.707。  
3. 未完成的 w12 / f30 不得混入主报；若恢复须重新 fuse + audit。  
4. 用 V3/V4 强分数校准的 Bayes 约 0.706；继续堆同质多样性的期望收益已在噪声带。  
5. 下一轮若声称突破，必须同时展示：嵌套均值、选择乐观、打乱标签、以及更新后的 Bayes / 邻域同标率。  
6. **基线三臂溯源**：`cat_d5/d6/alt` 的 per-seed parts 已在 `artifacts/worlds10/`（git 跟踪，与 arm bag 逐元素一致）；`artifacts/v4/arm_cat_{d5,d6,alt}.json` 已补齐 parts 清单。
