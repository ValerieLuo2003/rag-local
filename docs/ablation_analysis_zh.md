# 消融实验分析：Chunk Size、Top-k 与检索器对比

## 1. 实验设置

数据集：BEIR SciFact

评估文件：`eval/scifact_eval.jsonl`

评估问题数：300

Chunk overlap：100 个字符

对比的检索方法：

- BM25：关键词检索，偏精确词匹配。
- Embedding：向量语义检索，使用 `sentence-transformers/all-MiniLM-L6-v2`。
- Hybrid：BM25 + Embedding，用 RRF 做排序融合。

对比的 chunk size：

- 600
- 1200
- 1800

对比的 top-k：

- 3
- 5
- 10

评估指标：

- Recall@k：标准相关文档是否出现在 top-k 检索结果中。
- MRR@k：第一个标准相关文档排得是否靠前。

## 2. 完整结果

| 检索器 | Chunk Size | Top-k | Recall | MRR |
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

### 结论 1：Top-k 越大，Recall 通常越高

在所有检索器和 chunk size 下，top-k 从 3 增加到 5、10 时，Recall 基本都会提升。

以 `chunk_size=1800` 为例：

| 检索器 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.7100 | 0.7400 | 0.7967 |
| Embedding | 0.6800 | 0.7633 | 0.8133 |
| Hybrid | 0.7333 | 0.7700 | 0.8333 |

解释：

- top-k 越大，系统越有机会把标准相关文档召回。
- 但真实 RAG 生成时，top-k 越大也会带来更多 token、更高成本和更多噪声。

实际建议：

- 检索评估可以用 `top_k=10` 看召回上限。
- 调用 LLM 生成答案时建议先用 `top_k=3` 或 `top_k=5` 控制成本。

### 结论 2：Hybrid 整体最稳定

Hybrid 在大多数设置下都优于单独 BM25 或单独 Embedding。

在 `chunk_size=1800, top_k=10` 时：

| 检索器 | Recall@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.7967 | 0.6378 |
| Embedding | 0.8133 | 0.6088 |
| Hybrid | 0.8333 | 0.6688 |

解释：

- BM25 对术语、数字、关键词匹配更强。
- Embedding 对语义相近、表达不同的问题更强。
- RRF 融合只依赖排名，不强行比较 BM25 分数和向量相似度分数，因此比较稳。

面试表达：

> BM25 和 embedding retrieval 的优势互补。SciFact 实验中，Hybrid 在 Recall@10 和 MRR@10 上整体优于单一路线，说明关键词信号和语义信号都对科学文献检索有价值。

### 结论 3：这个数据集上较大的 chunk 更好

BM25 的效果随 chunk size 增大而提升：

| Chunk Size | Recall@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7700 | 0.5953 |
| 1200 | 0.7733 | 0.6127 |
| 1800 | 0.7967 | 0.6378 |

Hybrid 在 `chunk_size=1800` 时 MRR 最好：

| Chunk Size | Recall@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.8333 | 0.6336 |
| 1200 | 0.8300 | 0.6414 |
| 1800 | 0.8333 | 0.6688 |

解释：

- SciFact 文档大多是较短的论文摘要。
- 较大的 chunk 更容易保留完整摘要语义。
- 所以在这个数据集里，`chunk_size=1800` 比较合适。

注意：

- 这不代表所有 RAG 场景都应该用大 chunk。
- 对企业长文档来说，过大的 chunk 会引入噪声，也会增加 LLM token 成本。

### 结论 4：Embedding 提升召回，但排序不一定最好

Embedding 的 Recall@10 随 chunk size 增大而提升：

| Chunk Size | Recall@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7933 | 0.6130 |
| 1200 | 0.8067 | 0.6009 |
| 1800 | 0.8133 | 0.6088 |

解释：

- Embedding 检索更容易找到语义相关文档。
- 但标准相关文档不一定排在最前面。
- 所以 embedding 更像是提升 recall 的召回器，最终排序还需要 hybrid 或 rerank。

这也是加入 cross-encoder reranker 的动机。

## 4. 当前最佳配置

不使用 rerank 时，当前最佳配置是：

```text
retriever=hybrid
chunk_size=1800
top_k=10
Recall@10=0.8333
MRR@10=0.6688
```

之前跑过的 rerank 配置是：

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1200
top_k=10
Recall@10=0.8500
MRR@10=0.6594
```

注意：

- rerank 在 `chunk_size=1200` 下 Recall 更高。
- hybrid 在 `chunk_size=1800` 下 MRR 更高。
- 这两个还不能直接下最终结论，因为 rerank 还没有在 `chunk_size=1800` 下测试。

下一步推荐实验：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --retriever rerank --rerank-base hybrid --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1800 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1800.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --progress-every 50
```

## 5. 生成阶段建议

检索评估和 LLM 生成不一定用同一个 top-k。

检索评估：

```text
top_k=10
```

看的是系统能不能召回标准证据。

LLM 生成：

```text
top_k=3 或 top_k=5
max_context_chars=2500
max_output_tokens=300
```

看的是在控制成本的情况下，让模型基于少量高质量证据回答。

推荐生成配置：

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1800
top_k=3
hybrid_candidate_k=30
rerank_candidate_k=30
max_context_chars=2500
max_output_tokens=300
```

## 6. 失败案例分析方向

之前这条问题：

```text
0-dimensional biomaterials show inductive properties.
```

检索没有找到标准相关文档，导致 DeepSeek 正确拒答。

这个现象很重要：

> RAG 的答案质量受到检索质量上限约束。如果正确证据没有被召回，安全的 LLM 应该拒答，而不是编造答案。

后续可以优化：

- Query rewrite：把科学 claim 改写成更适合检索的 query。
- 领域 embedding：换成 biomedical/scientific embedding model。
- 领域 reranker：换成科学文献检索场景更适合的 reranker。
- 更大的候选集：例如 hybrid top-100 后再 rerank。
- 更细的 metadata：标题、摘要、文档 id 分字段保存和检索。

## 7. 面试复述版

可以这样说：

> 我在 BEIR SciFact 上做了检索消融实验，对比了 BM25、embedding retrieval 和 hybrid retrieval 在不同 chunk size 和 top-k 下的表现。实验发现，top-k 增大可以提升 Recall，但会增加上下文成本；Hybrid 整体最稳定，因为它融合了 BM25 的关键词匹配和 embedding 的语义匹配。在 SciFact 这种短摘要数据上，chunk size 1800 表现较好，说明完整摘要上下文对科学文献检索比较重要。当前非 rerank 最优配置是 hybrid + chunk size 1800 + top-k 10，Recall@10 达到 0.8333，MRR@10 达到 0.6688。下一步可以测试 chunk size 1800 下的 hybrid + cross-encoder rerank，看是否进一步提升排序质量。

## 8. 阶段性项目复盘

到目前为止，这个项目已经不只是一个“调用 LLM 的 demo”，而是一个比较完整的 RAG 应用算法项目雏形。它包含了文档加载、chunk 切分、BM25 检索、embedding 语义检索、hybrid retrieval、cross-encoder rerank、检索评估、消融实验、LLM 生成、引用检查和拒答逻辑。

当前项目主链路可以概括为：

```text
文档集合
-> 文档解析
-> chunk 切分
-> BM25 / embedding 建索引
-> query 检索候选 chunk
-> hybrid 融合召回
-> cross-encoder rerank
-> 选择 top-k 证据
-> 拼接 prompt
-> LLM 基于证据回答
-> 引用检查 / 拒答检查
```

### 8.1 已完成内容

目前已经完成的核心模块：

- 基础 RAG 流程：支持从本地文档中检索证据，并基于证据生成回答。
- BM25 检索：用于关键词、专有名词、编号类问题的精确匹配。
- Embedding 检索：使用 sentence-transformers 将 query 和 chunk 编码成向量，通过向量相似度做语义召回。
- Hybrid retrieval：使用 RRF 融合 BM25 和 embedding 的排序结果，提高整体召回稳定性。
- Cross-encoder rerank：对召回候选做更精细的 query-document 相关性排序。
- 检索评估：使用 Recall@k 和 MRR 评估相关文档是否被召回、是否排在靠前位置。
- 消融实验：对比不同 chunk size、top-k、retriever 对结果的影响。
- LLM 接入：已经支持 DeepSeek 这类 OpenAI-compatible Chat Completions API。
- 引用与拒答：要求模型输出引用；如果证据不足，系统应该拒答而不是编造。

这几个点已经能支撑一段比较完整的项目介绍。

### 8.2 当前实验结论

从当前 SciFact 消融实验看，主要结论是：

- `top_k` 增大通常会提高 Recall，但会增加后续 LLM 的上下文 token 成本。
- Hybrid retrieval 整体比单独 BM25 或单独 embedding 更稳定。
- 在 SciFact 这种论文摘要较短的数据集上，`chunk_size=1800` 表现较好，因为它更容易保留完整摘要语义。
- Embedding 检索能提高语义召回能力，但排序质量不一定总是最好，所以需要 hybrid 和 rerank。
- RAG 的最终回答质量受检索上限约束。如果正确证据没有被召回，LLM 再强也很难给出可靠答案。

当前非 rerank 最优结果：

```text
retriever=hybrid
chunk_size=1800
top_k=10
Recall@10=0.8333
MRR@10=0.6688
```

之前 rerank 结果：

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1200
top_k=10
Recall@10=0.8500
MRR@10=0.6594
```

所以后续最自然的实验是：测试 `chunk_size=1800` 下的 `hybrid + rerank`，看看能不能同时保持较高 Recall 和更好的排序质量。

### 8.3 LLM 调用阶段的观察

DeepSeek API 已经可以正常调用。用样例文档提问“RAG 的主流程是什么？”时，系统能够基于检索到的证据生成回答，并输出引用。

但是在 SciFact 的事实判断问题中，如果检索到的证据不相关，LLM 会输出“无法确定”。这不是坏结果，反而说明拒答策略生效了。

这个现象可以在面试中这样解释：

> 我把 RAG 拆成检索层和生成层来看。生成效果不好时，不一定是 LLM 的问题，也可能是检索没有召回正确证据。因此我会先看 evidence 是否命中，再看回答是否忠实于 evidence。对于证据不足的问题，系统应该拒答，这比强行生成更安全。

### 8.4 当前项目的亮点

这个项目目前比较适合写进简历的点：

- 不是只调用 API，而是实现了完整 RAG 检索链路。
- 有 BM25、embedding、hybrid、rerank 多种检索策略对比。
- 有离线评估指标，而不是只靠主观观察。
- 有 chunk size、top-k、retriever 的消融实验。
- 有引用输出和拒答逻辑，体现了幻觉控制意识。
- 接入了真实 LLM API，验证了端到端问答流程。

简历表达可以写：

```text
构建基于 RAG 的本地知识库问答系统，支持文档解析、chunk 切分、BM25/embedding/hybrid retrieval、cross-encoder rerank、基于引用的答案生成和低置信拒答。基于 BEIR SciFact 构造检索评估流程，对比不同 chunk size、top-k 和检索策略对 Recall@k、MRR 的影响；实验中 hybrid retrieval 在 Recall@10 和 MRR@10 上整体优于单一路线。
```

### 8.5 当前不足

当前项目还可以继续加强的地方：

- 生成评估还比较粗，目前主要看引用是否存在、是否拒答，还没有系统评估 answer correctness。
- SciFact 是公开 benchmark，格式比较规整，和真实企业知识库还有差距。
- 目前 chunk 策略还是固定长度切分，还没有做按标题、段落、语义结构切分。
- embedding 和 reranker 使用的是通用模型，未必最适合科学文献场景。
- 还没有做 query rewrite，也没有做多轮对话或 agent/tool use。
- 还没有可视化界面，展示上偏命令行。

这些不足不是问题，反而可以作为后续优化方向和面试追问的回答材料。

## 9. 后续优化优先级

如果继续优化，我建议不要一次加太多功能。优先做下面这些最有性价比的内容。

### 优先级 1：补充 rerank 消融

先测：

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1800
top_k=10
```

目的：验证当前最优 chunk size 下，加 rerank 是否进一步提升 MRR 或 Recall。

这一步很适合作为“下一轮实验”，因为它直接接在当前消融结论后面。

### 优先级 2：做错误分析

挑出几类 case：

- BM25 命中但 embedding 没命中。
- Embedding 命中但 BM25 没命中。
- Hybrid 命中但单路没命中。
- 所有方法都没命中。
- 检索命中但 LLM 回答不理想。

错误分析比继续盲目调参数更有价值，因为它能说明系统失败在哪里。

面试里可以说：

> 我不是只看总体指标，还会看失败样例，把问题拆成关键词匹配失败、语义召回失败、排序失败和生成忠实性失败，再决定下一步优化。

### 优先级 3：做小规模生成评估集

可以先不用大规模调用 API，手工挑 10-20 条问题即可：

- 证据明确支持的问题。
- 证据明确反对的问题。
- 证据不足、应该拒答的问题。
- 检索容易混淆的问题。

记录：

```text
question
gold_doc
retrieved_doc_hit
answer
has_citation
citation_valid
should_refuse
actually_refused
human_correctness
```

这一步能把项目从“检索项目”推进到“端到端 RAG 项目”。

### 优先级 4：优化 chunk 策略

当前是固定长度 chunk，可以继续加：

- 按段落切分。
- 按标题切分。
- sentence-aware splitting。
- 对标题和正文保留 metadata。

但这一步可以稍后做，因为 SciFact 文档本身比较短，chunk 策略的收益可能不如企业长文档明显。

### 优先级 5：做一个轻量界面

最后可以用 Streamlit 做一个很简单的界面：

- 左侧上传文档或选择数据集。
- 中间输入问题。
- 右侧显示答案、引用证据、检索分数。
- 展示当前 retriever、top-k、chunk size 配置。

界面不是算法核心，但展示项目时会更直观。

## 10. 当前阶段建议

现在最推荐的节奏是：

```text
1. 先把当前复盘文档整理好。
2. 再补一个 rerank chunk_size=1800 的实验。
3. 再挑 10-20 条问题做小规模生成评估。
4. 最后写项目总 README 和简历描述。
```

不要急着继续堆 agent、LoRA、复杂工具调用。对 LLM 应用算法岗来说，当前最该强化的是：

- 检索为什么这样设计。
- 实验为什么这样对比。
- 指标说明了什么。
- 失败案例暴露了什么问题。
- 下一步优化为什么这么做。

把这些讲清楚，比功能多但讲不清楚更有价值。
