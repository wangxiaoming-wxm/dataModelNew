# artifacts/v4max3pro

## 最终配方需要的文件

| 文件 | 用途 |
|---|---|
| `plus_strong.npz` | 最终第 4 臂 |
| `noxb10.npz` | 最终第 5 臂（8 seed） |
| `recipe_report.json` | 由 `src4/build_submission_v4max3pro.py` 生成 |
| `status_report.json` | 摘要 |

## 不要用这些做正式复现

中间 `part_*.npz` 不应提交。探索臂产物已从本目录移除。

校验：

```bash
python3 src4/build_submission_v4max3pro.py --check
```
