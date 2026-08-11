# 待接入参考（云端不可读本机路径）

用户本机路径（Cloud Agent **无法访问** `/Volumes/pssd/...`）：

1. `.../特征设计文档_best_v1.md`
2. `.../explore_best.py`

请把这两份文件**上传到仓库**或复制到：

`refs_incoming/特征设计文档_best_v1.md`  
`refs_incoming/explore_best.py`

接入后将：
- 按文档做折内 FE 探针臂
- 用 `explore_best.py` 的筛选逻辑做单臂门禁（AUC / corr）
- 仅当过策略门禁才进 `max(rank)` 四臂融合
