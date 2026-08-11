# SUPER714 方案

## 1. 当前结论

SUPER714 以线上 AUC `0.71453` 的 best_v1 为安全基座，并为折内
`TE(source × days_bin10)` 保留第三臂。只有第三臂同时通过强度、差异性和融合增益三项门槛，
主提交才会升级为 max3；否则交付已验证的 max2，避免为了本地分数堆叠高相关臂。

这里的“期望超越”是有门槛的实验假设，不是对未知榜单成绩的保证。

## 2. best_v1 基座

| 组件 | 配方 | 已验证 OOF AUC |
|---|---|---:|
| main | cond_r 世界；RMSE；Ordered；depth 5；800 trees；l2 10；8 seeds × 3 bags | 0.69992 |
| alt | rate 世界；RMSE；Plain；depth 6；800 trees；l2 6；8 seeds × 3 bags | 0.69770 |
| max2 | `max(rank(main), rank(alt))` | **0.70128** |

线上锚点为 `0.71453`。预计算文件从 `real714_pkg/v1_best714` 原样复制，
SHA-256 固化在 `artifacts/super714/manifest.json`，可秒级验收。

## 3. 为什么第三臂可能提升

main 的连续 `ratio=days/cond_r` 描述个体相对车群状态；候选 TE 捕捉离散
“source × 天数段”的折内历史索赔率。两者归纳偏置不同，理论上可能产生互补排序。
但历史实验也表明多臂 max 会放大 OOF 选择偏差，因此使用预先固定的入场条件：

1. TE 臂 OOF AUC `> 0.697`；
2. TE 臂与 main 的相关系数 `< 0.90`；
3. `max3 OOF >= max2 OOF + 0.001`。

任一条件不满足即保留 max2。门槛在训练前写定，不根据测试集或榜单反馈调整。

## 4. 复现

```bash
pip install -r requirements.txt

# 秒级验收（不训练）
bash run_super714.sh --verify

# 通路检查；输出带 _smoke 后缀，不覆盖主交付
bash run_super714.sh --smoke

# 完整训练
bash run_super714.sh
```

可通过 `DATA_DIR` 或 `--data-dir` 指向含 `train.csv/test.csv` 的目录。

## 5. 产物

- 主提交：`submissions/submission_super714.csv`
- 预计算 OOF：`artifacts/super714/best_v1_oof.npy`
- 预计算测试预测：`artifacts/super714/best_v1_test.npy`
- 秒级验收：`src_super/verify_super714.py`
- 完整训练入口：`src_super/train_super714.py`

当前主提交来自已验证 best_v1 fuse 产物；完整训练完成并通过门槛后才允许覆盖。

## 6. 诚实协议

- 只使用仓库给定数据，不使用外部数据或伪标签。
- 不读取、推断或利用测试标签。
- TE 的分桶、统计量和平滑映射必须在对应训练折内拟合。
- OOF 仅用于预先声明的门槛判断；不进行大规模臂搜索或榜单闭环。
- 冒烟产物不会覆盖主提交。
