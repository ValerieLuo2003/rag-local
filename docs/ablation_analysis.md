# 消融实验分析：Chunk Size、Top-k 与检索器对比

## 1. 实验设置

- 数据集：BEIR SciFact
- 评测集：`eval/scifact_eval.jsonl`
- 查询数：300
- Chunk overlap：100 个字符

对比的检索器：

- BM25：基于关键词的稀疏检索；
- Embedding：使用 `sentence-transformers/all-MiniLM-L6-v2` 的稠密语义检索；
- Hybrid：使用 RRF 融合 BM25 与 Embedding 的排序结果。

对比的 chunk size：

- 600
- 1200
- 1800

对比的 top-k：

- 3
- 5
- 10

历史实验使用的指标：

- Hit@K：Top-K 检索结果中是否至少包含一个 gold source；
- MRR@K：第一个 gold source 在 Top-K 中出现位置的倒数。

注意：旧实验记录曾把“Top-K 是否命中任意一个相关来源”写成 Recall。该指标严格来说是 Hit@K，因此下表已使用正确名称。真正的 Recall@K 需要计算“找回的不同相关来源数 / 全部相关来源数”。

## 2. 完整结果

| 检索器 | Chunk Size | Top-k | Hit@K | MRR@K |
|---|---:|---:|---:|---:|
| BM25 | 600 | 3 | 0.6533 | 0.5750 |
| BM25 | 600 | 5 | 0.6967 | 0.5850 |
| BM25 | 600 | 10 | 0.7700 | 0.5953 |
| Embedding | 600 | 3 | 0.6867 | 0.5906 |
| Embedding | 600 | 5 | 0.7633 | 0.6086 |
| Embedding | 600 | 10 | 0.7933 | 0.6130 |
| Hybrid | 600 | 3 | 0.6933 | 0.6083 |
| Hybrid | 600 | 5 | 0.7600 | 0.6233 |
| Hybrid | 600 | 10 | 0.8333 | 0.6336 |
| BM25 | 1200 | 3 | 0.6867 | 0.5972 |
| BM25 | 1200 | 5 | 0.7267 | 0.6061 |
| BM25 | 1200 | 10 | 0.7733 | 0.6127 |
| Embedding | 1200 | 3 | 0.6600 | 0.5717 |
| Embedding | 1200 | 5 | 0.7533 | 0.5933 |
| Embedding | 1200 | 10 | 0.8067 | 0.6009 |
| Hybrid | 1200 | 3 | 0.7133 | 0.6233 |
| Hybrid | 1200 | 5 | 0.7500 | 0.6315 |
| Hybrid | 1200 | 10 | 0.8300 | 0.6414 |
| BM25 | 1800 | 3 | 0.7100 | 0.6228 |
| BM25 | 1800 | 5 | 0.7400 | 0.6294 |
| BM25 | 1800 | 10 | 0.7967 | 0.6378 |
| Embedding | 1800 | 3 | 0.6800 | 0.5833 |
| Embedding | 1800 | 5 | 0.7633 | 0.6023 |
| Embedding | 1800 | 10 | 0.8133 | 0.6088 |
| Hybrid | 1800 | 3 | 0.7333 | 0.6522 |
| Hybrid | 1800 | 5 | 0.7700 | 0.6606 |
| Hybrid | 1800 | 10 | 0.8333 | 0.6688 |

## 3. 主要结论

### 结论 1：增大 top-k 能提高命中率，但会增加上下文成本

对所有检索器和 chunk size，top-k 从 3 增至 5、再增至 10 时，Hit@K 基本都会提高。

以 `chunk_size=1800` 为例：

| 检索器 | Hit@3 | Hit@5 | Hit@10 |
|---|---:|---:|---:|
| BM25 | 0.7100 | 0.7400 | 0.7967 |
| Embedding | 0.6800 | 0.7633 | 0.8133 |
| Hybrid | 0.7333 | 0.7700 | 0.8333 |

解释：

- 更大的 top-k 给检索器更多机会召回 gold evidence；
- 但在真实 RAG 系统中，更大的 top-k 也意味着更多 prompt token、更多噪声、更高成本，并可能降低生成质量。

实践建议：

- 检索评测使用 `top_k=10`；
- LLM 回答生成使用 `top_k=3` 或 `top_k=5`，控制 token 成本与噪声。

### 结论 2：Hybrid 是整体最稳定的检索器

在不同 chunk size 和 top-k 下，Hybrid 通常是表现最稳定的方法。

当 `chunk_size=1800, top_k=10` 时：

| 检索器 | Hit@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.7967 | 0.6378 |
| Embedding | 0.8133 | 0.6088 |
| Hybrid | 0.8333 | 0.6688 |

原因：

- BM25 擅长匹配科学术语、数字和固定短语；
- Embedding 能处理查询与文档措辞不同但语义接近的情况；
- RRF 不要求 BM25 分数与余弦相似度位于同一尺度，能直接融合两路排序。

面试时可以表述为：

> BM25 与向量检索具有互补优势。SciFact 的历史实验中，Hybrid 同时提高了 Hit@10 和 MRR@10，说明科学文献检索既需要精确词法信号，也需要语义信号。

### 结论 3：在 SciFact 上，较大的 chunk 对 BM25 和 Hybrid 更有利

BM25 在 `top_k=10` 时：

| Chunk Size | Hit@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7700 | 0.5953 |
| 1200 | 0.7733 | 0.6127 |
| 1800 | 0.7967 | 0.6378 |

Hybrid 在 `top_k=10` 时：

| Chunk Size | Hit@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.8333 | 0.6336 |
| 1200 | 0.8300 | 0.6414 |
| 1800 | 0.8333 | 0.6688 |

解释：

- SciFact 文档主要是较短的科学摘要；
- 较大的 chunk 更容易保留完整的摘要级上下文；
- `chunk_size=1800` 能减少相关证据被拆散的情况。

局限：

- 这不代表 chunk 越大越好；
- 对企业长文档或标准文档，过大的 chunk 会引入噪声并增加 LLM 上下文成本；
- 密码学语料已经改用章节/页码感知切分，不能直接套用 SciFact 的最优值。

### 结论 4：Embedding 提高命中覆盖，但首个相关结果不一定更靠前

Embedding 的历史结果：

| Chunk Size | Hit@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7933 | 0.6130 |
| 1200 | 0.8067 | 0.6009 |
| 1800 | 0.8133 | 0.6088 |

解释：

- 稠密检索擅长找到语义相关证据；
- 但 gold evidence 不一定排在最前；
- 因此 Embedding 的 Hit@10 会提高，MRR 却不一定优于 BM25 或 Hybrid。

这也是引入 reranker 的原因：

> Embedding 适合扩大候选覆盖，Cross-Encoder reranker 负责改善最终 Top-K 的排序质量。

## 4. 最佳配置

### 最佳非 Rerank 配置

```text
retriever=hybrid
chunk_size=1800
top_k=10
Hit@10=0.8333
MRR@10=0.6688
```

### 历史 Rerank 最佳配置

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1200
top_k=10
Hit@10=0.8500
MRR@10=0.6594
```

对比：

- Rerank 在 chunk size 1200 时把 Hit@10 提高到 0.8500；
- Hybrid 在 chunk size 1800 时无需 rerank 即得到更高的 MRR@10：0.6688；
- 两组结果不能完全直接比较，因为 rerank 尚未在 chunk size 1800 上复算。

建议补充实验：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval `
  --retriever rerank `
  --rerank-base hybrid `
  --docs data/scifact_docs `
  --eval-file eval/scifact_eval.jsonl `
  --top-k 10 `
  --chunk-size 1800 `
  --chunk-overlap 100 `
  --embedding-cache vector_store/scifact_all-MiniLM_chunk1800.npz `
  --model-cache-dir model_cache `
  --hybrid-candidate-k 50 `
  --rerank-candidate-k 50 `
  --progress-every 50
```

## 5. 推荐设置

检索评测：

```text
retriever=hybrid or rerank
chunk_size=1800
top_k=10
chunk_overlap=100
```

控制 API 成本的 LLM 生成：

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1200 or 1800
top_k=3
hybrid_candidate_k=30
rerank_candidate_k=30
max_context_chars=2500
max_output_tokens=300
```

原因：

- 检索评测可以使用较大的 top-k 测量候选覆盖；
- 生成阶段应缩小最终 top-k，减少 token 成本和无关上下文。

## 6. 失败案例分析

早期查询：

```text
0-dimensional biomaterials show inductive properties.
```

失败原因是检索结果没有包含 gold source。DeepSeek 选择拒答，说明生成层的安全行为是合理的，真正的问题发生在检索阶段。

这个案例体现了一个重要结论：

> RAG 回答质量受检索质量上限约束。正确证据没有被召回时，安全的 LLM 应当拒答，而不是编造答案。

可能的改进：

- Query rewriting：把科学主张改写为更适合检索的查询；
- 领域 Embedding：使用生物医学或科学文献向量模型；
- 领域 Reranker：使用科学检索数据训练的重排模型；
- 多阶段检索：将 rerank 前候选集从 Top-50 扩大到 Top-100；
- 元数据感知检索：分别保留标题、摘要和文档 ID 字段。

## 7. 面试讲解摘要

一分钟版本：

> 我在 BEIR SciFact 上比较了 BM25、向量检索和 Hybrid，并对 chunk size 与 top-k 做了消融。历史结果显示，增大 top-k 能提高 Hit@K，但会增加上下文成本；Hybrid 最稳定，因为它结合了 BM25 的词法匹配和 Embedding 的语义匹配。SciFact 文档主要是短摘要，因此 1800 字符的较大 chunk 能保留更完整的证据。最佳非重排配置是 Hybrid、chunk size 1800、top-k 10，Hit@10 为 0.8333，MRR@10 为 0.6688。下一步应补测 chunk size 1800 下的 Hybrid + Cross-Encoder Rerank，并用当前严格指标重新计算 Recall@K 与 nDCG@K。

更完整的中文讲解与生成评测设计见 [ablation_analysis_zh.md](ablation_analysis_zh.md)。
