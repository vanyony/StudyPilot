# 本地 Embedding 选型记录

## 固定配置

- 模型：`BAAI/bge-small-zh-v1.5`
- Revision：`4e17e244a0fb63bfb78fca8fcf95079fcc664f5c`
- 参数量：官方模型卡标注约 24M
- 向量维度：运行时读取为 512
- License：MIT
- 查询指令：`为这个句子生成表示以用于检索相关文章：`
- Passage：`section + 原文`，不添加查询指令
- 向量：L2 归一化，点积等价于余弦相似度
- 设备：CPU
- 模型缓存：`D:\ModelCache\huggingface`，不提交仓库
- Block 向量缓存：`D:\ModelCache\studypilot-embeddings`，按模型、revision、section 和内容哈希寻址

官方资料：

- [BAAI 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [固定 revision](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/4e17e244a0fb63bfb78fca8fcf95079fcc664f5c)
- [FlagEmbedding 官方用法](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/inference/embedder/README.md)

官方说明短查询到长文本检索应给 query 添加指令、passage 不添加，并推荐归一化后使用内积；实现遵循该方式。

## 本机资源记录

- Hugging Face 模型缓存约 96.4 MB；
- 当前评测 Block 的 JSON 向量缓存约 3.1 MB（缓存键升级后可能暂时保留旧文件）；
- 安装 CPU PyTorch、sentence-transformers 及全部项目依赖后，整个 `.venv` 约 1.05 GB；这是环境总大小，不是模型自身大小；
- 首次包含下载的全流程约 64.7 秒；模型已下载、重新计算新缓存键后的完整三路评测约 43.0 秒。

