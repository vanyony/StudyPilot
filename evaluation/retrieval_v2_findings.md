# 30 条冻结评测集检索对比

数据集为 `probability-bm25-v1`，包含 30 条人工核对 query，覆盖九个章节。三路检索使用同一语料、query 和 gold。

| 检索方法 | Recall@5 | 术语查询 | 语义改写 |
|---|---:|---:|---:|
| 正文 BM25 | 22/30 = 73.3% | 12/15 | 10/15 |
| Title-aware BM25（title weight=2） | 26/30 = 86.7% | 15/15 | 11/15 |
| BGE Dense | 23/30 = 76.7% | 14/15 | 9/15 |
| RRF Hybrid（k=60） | 25/30 = 83.3% | 14/15 | 11/15 |

固定配置：BM25 `k1=1.5, b=0.75`；section tokens 复制 2 次后与正文 tokens 合并；Dense 使用 `BAAI/bge-small-zh-v1.5` 固定 revision；RRF 使用 `k=60`、每路最多 50 个候选。

## 观察

- 标题加权将正文 BM25 从 22/30 提升到 26/30，15 条术语查询全部命中；
- Dense 对语义改写的表现没有超过标题加权 BM25；
- RRF 恢复了 Dense 的部分遗漏，但总体仍低于最强单路；
- `semantic-02`、`semantic-08`、`semantic-10`、`semantic-11` 在三种升级方案中均未命中，适合作为后续 query expansion、rerank 或数据表示改进的固定回归样本。

结论：当前可复现结果支持“标题字段加权改善召回”，不支持“混合检索必然优于单路检索”。
