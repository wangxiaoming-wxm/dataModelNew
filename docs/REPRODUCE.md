# 复现说明（task-am40）

## 目标

验收 **AM40** 提交：在冻结 best_v1 双臂上复算融合，核对 OOF 锚点与提交文件哈希。

**不需要重训 CatBoost。**

## 环境

```bash
git clone https://github.com/wangxiaoming-wxm/dataModelNew.git
cd dataModelNew
git checkout task-am40
python3 -m pip install -r requirements.txt
```

数据已在 `data/`；冻结臂在 `artifacts/super714/best_v1_{oof,test}.npy`。

## 验收（秒级）

```bash
# 重新生成并校验
bash run_am40.sh

# 或只校验仓库内提交
bash run_am40.sh --verify
```

通过标准：

1. 打印 `PASS: AM40 融合验收通过（已超过 W62）`
2. OOF = **0.70181135**（相对 W62 +0.00021769）
3. `submissions/submission_am40.csv` 的 SHA-256 =
   `de0d337f02873f8b429c0518d3dadcbb34fdbd371e9707193dc2d1fbec49e2a9`
4. 与 `submission_w62.csv` 哈希不同

可选：`bash run_w62.sh --verify` 复核同臂 W62 基线。

## 若需从零重训双臂（非本分支必需）

见 `docs/SOLUTION.md` / `bash run_super714.sh`（耗时长，且本分支交付不依赖重训）。
