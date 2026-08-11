# SUPER714-Plus（可提交差异化版）

相对已公开的 best_v1 / `submission_super714.csv`（线上 0.71453），本版**刻意不同**，供比赛另交。

## 与冠军的差异

| 维度 | best_v1（冠军） | SUPER714-Plus |
|---|---|---|
| main depth | Ordered **5** | Ordered **6** |
| alt l2 / 分箱 | l2=6，bins (7,13,25) | l2=**5**，bins **(6,12,24)** |
| 特征 | 单世界 | main **+rate**；alt **+ratio/cond_r** |
| seeds × bags × trees | 8 × 3 × 800 | **10 × 4 × 1000** |
| seed 起点 | 2026 | **3100** |
| 提交文件 | `submission_super714.csv` | `submission_super714_plus.csv` |

融合仍为 `max(rank(main), rank(alt))`。

## 复现

```bash
pip install -r requirements.txt

# 冒烟
bash run_super714_plus.sh --smoke

# 完整训练（约数小时 @ 4 核）
bash run_super714_plus.sh
```

产物：

- `submissions/submission_super714_plus.csv`
- `artifacts/super714_plus/plus_{oof,test}.npy`
- `artifacts/super714_plus/metrics.json`（含相对冠军的 ΔOOF / test Spearman）

## 设计理由（简）

1. **不改信号轴**：仍用 cond_r / rate 双世界（历史唯一有效轴）。  
2. **用更多 bag/seed/树做稳定化**：bagging 是 D→best_v1 已验证增益维。  
3. **轻度跨世界连续特征**：增加臂内信息，但不引入已证伪的 TE / 第三世界。  
4. **改分箱与超参**：保证提交哈希与冠军不同，避免“交同一份”。  
5. **避开死路**：不加高相关第三臂、不用监督 stack、不堆 10+ 同族臂。
