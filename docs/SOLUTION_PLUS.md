# SUPER714-Plus（可提交差异化版）

相对已公开的 best_v1 / `submission_super714.csv`（线上 0.71453），本版**刻意不同**，供比赛另交。

**当前线上最强锚点**：W62（同 best_v1 双臂，`0.62*main+0.38*alt`）→ **0.71503**。  
Plus 完成后除 max2 外，**必须**再出 W62 式加权提交（见 `src_super/fuse_plus_weights.py`）。

## 与冠军的差异

| 维度 | best_v1（冠军） | SUPER714-Plus |
|---|---|---|
| main depth | Ordered **5** | Ordered **6** |
| alt l2 / 分箱 | l2=6，bins (7,13,25) | l2=**5**，bins **(6,12,24)** |
| 特征 | 单世界 | main **+rate**；alt **+ratio/cond_r** |
| seeds × bags × trees | 8 × 3 × 800 | **10 × 4 × 1000** |
| seed 起点 | 2026 | **3100** |
| 融合 | max2（公开）/ W62（0.71503） | max2 + **w62** + OOF-wbest |
| 提交文件 | `submission_super714.csv` / `submission_w62.csv` | `submission_super714_plus*.csv` |

## 复现

```bash
pip install -r requirements.txt

# 冒烟
bash run_super714_plus.sh --smoke

# 完整训练（数小时～十余小时 @ 多核；主臂 Ordered d6 很慢）
bash run_super714_plus.sh

# 训练结束后若需单独重跑融合（含权重网格）
python3 -u src_super/fuse_plus_weights.py
```

产物：

- `submissions/submission_super714_plus.csv` — max2
- `submissions/submission_super714_plus_w62.csv` — 预注册 0.62/0.38
- `submissions/submission_super714_plus_wbest.csv` — OOF 网格最优权重
- `artifacts/super714_plus/plus_{oof,test}.npy`
- `artifacts/super714_plus/plus_{main,alt}.npy` — 臂级 checkpoint（新跑才有）
- `artifacts/super714_plus/metrics.json`

## 设计理由（简）

1. **不改信号轴**：仍用 cond_r / rate 双世界（历史唯一有效轴）。  
2. **用更多 bag/seed/树做稳定化**：bagging 是 D→best_v1 已验证增益维。  
3. **轻度跨世界连续特征**：增加臂内信息，但不引入已证伪的 TE / 第三世界。  
4. **改分箱与超参**：保证提交哈希与冠军不同，避免“交同一份”。  
5. **融合跟上 W62 证据**：嵌套里 max 常赢，但 best_v1 臂上加权已线上打到 **0.71503**；Plus 双轨输出。  
6. **避开死路**：不加高相关第三臂、不用监督 stack、不堆 10+ 同族臂。

## 状态

完整训练在 `cursor/super714-plus-edb2` 跑全量；日志 `logs/super714_plus_full.log`。  
当前跑次若在改代码前启动，结束后请再执行一次 `fuse_plus_weights.py` 生成 w62/wbest。
