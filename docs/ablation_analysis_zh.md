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

- Hit@k：top-k 检索结果中是否至少出现一个标准相关文档。
- MRR@k：第一个标准相关文档排得是否靠前。

## 2. 完整结果

> 指标口径说明：旧实验曾把“Top-K 是否命中任意一个相关来源”记作 Recall，
> 严格来说它是 Hit@K。下表已更正列名；真正的 Recall@K 需要计算找回的
> 不同相关来源数占全部相关来源数的比例。

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

### 结论 1：Top-k 越大，Hit@K 通常越高

在所有检索器和 chunk size 下，top-k 从 3 增加到 5、10 时，Hit@K 基本都会提升。

以 `chunk_size=1800` 为例：

| 检索器 | Hit@3 | Hit@5 | Hit@10 |
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

| 检索器 | Hit@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.7967 | 0.6378 |
| Embedding | 0.8133 | 0.6088 |
| Hybrid | 0.8333 | 0.6688 |

解释：

- BM25 对术语、数字、关键词匹配更强。
- Embedding 对语义相近、表达不同的问题更强。
- RRF 融合只依赖排名，不强行比较 BM25 分数和向量相似度分数，因此比较稳。

面试表达：

> BM25 和 embedding retrieval 的优势互补。历史 SciFact 实验中，Hybrid 在 Hit@10 和 MRR@10 上整体优于单一路线，说明关键词信号和语义信号都对科学文献检索有价值。

### 结论 3：这个数据集上较大的 chunk 更好

BM25 的效果随 chunk size 增大而提升：

| Chunk Size | Hit@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7700 | 0.5953 |
| 1200 | 0.7733 | 0.6127 |
| 1800 | 0.7967 | 0.6378 |

Hybrid 在 `chunk_size=1800` 时 MRR 最好：

| Chunk Size | Hit@10 | MRR@10 |
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

Embedding 的 Hit@10 随 chunk size 增大而提升：

| Chunk Size | Hit@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7933 | 0.6130 |
| 1200 | 0.8067 | 0.6009 |
| 1800 | 0.8133 | 0.6088 |

解释：

- Embedding 检索更容易找到语义相关文档。
- 但标准相关文档不一定排在最前面。
- 所以 embedding 更适合扩大候选覆盖，最终排序还需要 hybrid 或 rerank。

这也是加入 cross-encoder reranker 的动机。

## 4. Rerank 补充实验

在非 rerank 设置中，表现最好的配置为：

```text
retriever=hybrid
chunk_size=1800
top_k=10
Hit@10=0.8333
MRR@10=0.6688
```

随后补充了 `chunk_size=1800` 下的 `hybrid + cross-encoder rerank` 实验：

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1800
top_k=10
hybrid_candidate_k=50
rerank_candidate_k=50
Hit@10=0.8367
MRR@10=0.6588
```

对比结果如下：

| 配置 | Hit@10 | MRR@10 |
|---|---:|---:|
| hybrid, chunk=1800 | 0.8333 | 0.6688 |
| hybrid + rerank, chunk=1800 | 0.8367 | 0.6588 |

该结果表明，当前 cross-encoder reranker 在 `chunk_size=1800` 下仅小幅提升了 Hit@K，但没有提升 MRR，排序质量反而略有下降。因此，rerank 并不必然带来稳定收益，需要结合数据集、候选集质量、reranker 模型和 chunk 粒度共同评估。

可能原因包括：

- 当前 reranker 是通用英文检索模型，不一定适配 SciFact 的生物医学事实判断场景。
- base retriever 已经将部分标准证据排在较靠前位置，rerank 可能把表面相关但事实关系不匹配的候选提前。
- SciFact claim 通常较短，且包含大量专业实体、否定关系、比较关系和数字，通用 cross-encoder 对细粒度事实关系的排序不够稳定。
- `chunk_size=1800` 的文本较长，reranker 输入噪声增加，精排难度上升。

这一结果说明，RAG 系统优化不能仅凭模块堆叠判断效果，必须通过消融实验验证每个模块的实际贡献。

## 5. RAG 系统链路

当前项目已经实现一个可运行的 RAG 应用算法原型，包含文档加载、chunk 切分、BM25 检索、embedding 语义检索、hybrid retrieval、cross-encoder rerank、检索评估、LLM 生成、引用检查和拒答逻辑。

系统主链路如下：

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

核心模块如下：

- 基础 RAG 流程：从本地文档中检索证据，并基于证据生成回答。
- BM25 检索：用于关键词、专有名词和编号类问题的精确匹配。
- Embedding 检索：使用 sentence-transformers 将 query 和 chunk 编码成向量，通过向量相似度做语义召回。
- Hybrid retrieval：使用 RRF 融合 BM25 和 embedding 的排序结果，提高召回稳定性。
- Cross-encoder rerank：对召回候选做更精细的 query-document 相关性排序。
- 检索评估：使用 Hit@k、真正的 Recall@k、MRR 和 nDCG 评估相关文档覆盖与排序位置。
- LLM 生成：支持 OpenAI-compatible Chat Completions API，例如 DeepSeek。
- 引用与拒答：要求模型输出引用；当检索证据不足时返回无法确定，避免无依据生成。

## 6. 错误分析

错误分析采用“代码分桶 + 人工归因”的方式完成。代码负责找出不同检索器之间的命中差异，人工阅读典型 case 后总结失败原因。

错误分析文件位于：

```text
outputs/error_analysis_hybrid_vs_rerank_chunk1800_top10_first100.jsonl
```

每一行是一条 JSON 记录，主要字段如下：

| 字段 | 含义 |
|---|---|
| `bucket` | 样例所属类别 |
| `question` | SciFact claim / query |
| `relevant_sources` | 标准相关文档 |
| `top_a` | method_a 的 top 检索结果 |
| `top_b` | method_b 的 top 检索结果 |
| `preview` | 检索结果文本预览 |

本次分析比较 `hybrid` 与 `hybrid + rerank`，在前 100 条样例上的分桶结果如下：

```text
method_a=hybrid
method_b=rerank
top_k=10
chunk_size=1800
max_examples=100

a_only=6
b_only=2
both_hit=83
both_miss=9
```

分桶含义如下：

- `a_only`：hybrid 命中，但 rerank 未命中。
- `b_only`：rerank 命中，但 hybrid 未命中。
- `both_hit`：两个方法都命中。
- `both_miss`：两个方法都未命中。

该统计解释了 rerank 整体收益不明显的原因：rerank 补回了少量 hybrid 未命中的样例，但同时也丢失了更多 hybrid 原本已经命中的样例。

### 6.1 人工归因方法

人工分析单条 case 时，主要比较三类信息：

1. query 本身表达了什么事实关系。
2. 标准相关文档的标题和摘要是否直接覆盖该事实。
3. 检索结果返回的是标准证据、表面相似文档，还是主题相近但事实不匹配的文档。

常见失败原因包括：

- 关键词不重合：query 和标准文档之间缺少明显共享词，BM25 和通用 embedding 都难以召回。
- 表面词误导：返回文档包含相同实体或术语，但并不支持 query 中的具体关系。
- 否定或比较关系困难：query 中包含 `not`、`better`、`larger` 等判断，检索器可能只捕捉实体而忽略关系方向。
- 领域模型不适配：通用 embedding 或 reranker 对生物医学术语、缩写和细粒度事实关系不够敏感。
- chunk 噪声偏大：较长 chunk 保留了完整摘要，但也可能引入无关上下文，影响 reranker 判断。

### 6.2 典型样例

| Bucket | Question | 现象 | 可能原因 |
|---|---|---|---|
| `both_miss` | `0-dimensional biomaterials show inductive properties.` | 两个方法均未召回标准文档 `scifact_31715818.md` | 标准文档标题是纳米技术操控干细胞，query 与标题/摘要关键词不直接重合，存在语义跳跃 |
| `both_miss` | `Aspirin inhibits the production of PGE2.` | 返回结果集中在 PGE2、prostaglandin、aspirin 相关主题，但未命中标准文档 | 表面主题相关较强，但标准证据涉及 COX/PGE2/肿瘤免疫链路，事实关系更细 |
| `a_only` | `CRP is not predictive of postoperative mortality following CABG surgery.` | hybrid 在 top-5 命中标准文档，rerank 后标准文档跌出 top 结果 | reranker 更偏好标题中直接出现 CRP、CABG、mortality 的文档，忽略了标准文档中的决策模型和否定判断 |
| `a_only` | `Cells undergoing methionine restriction may activate miRNAs.` | hybrid 命中标准文档，rerank 被多个 miRNA 表面相关文档吸引 | reranker 对“methionine restriction”和“miRNA stress response”的组合关系不够稳定 |
| `b_only` | `Arterioles have a larger lumen diameter than venules.` | rerank 将标准文档提升进 top-k，hybrid 未命中 | rerank 能利用候选文本中的 arterioles、venules、angiogenesis 等上下文做二次排序 |

错误分析显示，当前系统的主要瓶颈不是单一模块失效，而是事实判断类 query 对“实体 + 关系 + 方向”的要求更高。关键词召回、语义召回和通用 rerank 都可能在细粒度关系上出错。

## 7. LLM 生成阶段

检索评估和 LLM 生成阶段的 top-k 目标不同。检索评估通常使用较大的 `top_k=10` 观察召回上限；生成阶段通常使用较小的 `top_k=3` 或 `top_k=5`，以控制 token 成本和上下文噪声。

当前生成阶段推荐配置如下：

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

DeepSeek API 已完成基本连通性验证。在样例文档问题“RAG 的主流程是什么？”中，系统能够基于检索证据生成答案，并输出引用。

在 SciFact 事实判断问题中，如果检索证据不相关，模型可能输出“无法确定”。这属于合理行为：RAG 的答案质量受到检索质量上限约束，当正确证据没有进入上下文时，拒答比无依据生成更安全。

## 8. 小规模生成评估设计

端到端生成评估需要真实 LLM API 才能得到最终回答。`mock` provider 只能验证检索、prompt、引用检查和拒答后处理流程，不能评估真实答案质量。

小规模生成评估不需要重新构造一套知识库。当前阶段可直接从 SciFact 中抽样构造小评估集，覆盖以下类型：

- 检索较容易命中的事实问题。
- 同一文档下方向相反或容易混淆的 claim。
- hybrid 命中但 rerank 未命中的样例。
- rerank 命中但 hybrid 未命中的样例。
- 两种方法都未命中的失败样例。

项目中已新增小规模评估文件：

```text
eval/generation_eval_small.jsonl
```

该文件包含 12 条问题，来源于 SciFact 原始评估集，并保留 `relevant_sources` 与 `review_group` 字段，便于后续人工复查。

该评估集已经完成一次 dry-run 验证，输出文件为：

```text
outputs/generation_eval_small_dryrun.jsonl
```

dry-run 不调用真实 LLM，只执行检索、rerank、prompt 构造和 token 估算。本次 dry-run 结果如下：

```text
examples=12
evidence_hit_rate=0.5000
citation_format_valid_rate=1.0000
refusal_rate=0.0000
```

该小集合刻意混合了容易命中的样例和检索失败样例，因此 `evidence_hit_rate=0.5000` 主要用于覆盖不同错误类型，不代表系统整体检索效果。

真实 DeepSeek API 评估输出文件为：

```text
outputs/generation_eval_small_deepseek.jsonl
```

真实生成评估结果如下：

```text
examples=12
evidence_hit_rate=0.5000
citation_format_valid_rate=0.9167
refusal_rate=0.4167
```

指标解释：

- `evidence_hit_rate=0.5000`：12 条问题中有 6 条检索到了标准相关文档。由于该小集合刻意包含失败样例和容易混淆样例，该指标主要用于观察端到端系统在不同检索质量下的表现。
- `citation_format_valid_rate=0.9167`：12 条回答中有 11 条引用编号格式有效。该指标只检查编号存在且不越界，不代表引用内容支持回答；Citation Correctness 和 Faithfulness 需要人工或 judge 复核。
- `refusal_rate=0.4167`：12 条回答中有 5 条触发拒答或被识别为拒答，说明模型在证据不足时具备一定谨慎性。

典型现象如下：

| 类型 | 样例 | 观察 |
|---|---|---|
| 检索命中且正常回答 | `ADAR1 binds to Dicer to cleave pre-miRNA.` | 标准证据 top-1 命中，模型基于证据回答并引用 `[1]` |
| 检索命中且正常回答 | `AIRE is expressed in some skin tumors.` | 标准证据 top-1 命中，模型能从证据中抽取 Aire 与 skin tumor keratinocytes 的关系 |
| 检索命中但拒答 | `ALDH1 expression is associated with better breast cancer outcomes.` | 证据实际支持 poorer prognosis，模型拒绝直接肯定 better outcomes |
| 检索失败且合理拒答 | `0-dimensional biomaterials show inductive properties.` | 未召回标准文档，模型输出无法确定 |
| 检索失败且合理拒答 | `Aspirin inhibits the production of PGE2.` | 检索结果主题相关但未命中标准证据，模型输出无法确定 |
| 检索失败但仍回答 | `CRP is not predictive of postoperative mortality following CABG surgery.` | 未命中 benchmark 标准文档，但检索到其他 CRP/CABG 相关文档，模型仍基于这些证据给出反驳 |
| 引用格式失败 | `Activation of PPM1D suppresses p53 function.` | 模型表达证据不足，但未按要求输出有效引用格式 |

该结果表明，端到端生成链路已经跑通，引用控制整体有效，模型具备一定拒答能力。但系统仍存在两个风险：

- 当检索未命中标准文档但返回了主题相关证据时，LLM 仍可能基于非标准证据生成答案，需要人工判断答案是否事实正确。
- SciFact 更接近事实验证任务，仅输出自然语言答案不够结构化；更合适的输出格式可能是 `SUPPORTS / REFUTES / NOT ENOUGH INFO + Answer + Citations`。

生成评估建议记录以下字段：

```text
question
review_group
relevant_sources
retrieved_sources
evidence_hit
answer
citations_valid
refused
human_correctness
human_note
```

其中 `human_correctness` 需要人工判断，原因是脚本目前只统计 evidence hit、citation validity 和 refusal rate，并不会自动判断答案事实正确性。

Dry-run 可用于检查证据和 prompt，不消耗 LLM API：

```powershell
$env:PYTHONPATH="src"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m rag_starter.eval_generation --dry-run --show-evidence --llm-provider mock --retriever rerank --rerank-base hybrid --docs data/scifact_docs --eval-file eval/generation_eval_small.jsonl --max-examples 12 --top-k 3 --chunk-size 1800 --chunk-overlap 100 --embedding-model model_cache_http\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embedding-cache vector_store\scifact_localpath_all-MiniLM_chunk1800.npz --model-cache-dir model_cache_http --hybrid-candidate-k 30 --rerank-candidate-k 30 --reranker-model model_cache_http\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8 --output outputs\generation_eval_small_dryrun.jsonl
```

真实 API 评估可使用 OpenAI-compatible provider，例如 DeepSeek：

```powershell
$env:PYTHONPATH="src"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m rag_starter.eval_generation --llm-provider openai-chat --api-key-env DEEPSEEK_API_KEY --openai-base-url https://api.deepseek.com --llm-model deepseek-v4-flash --thinking disabled --retriever rerank --rerank-base hybrid --docs data/scifact_docs --eval-file eval/generation_eval_small.jsonl --max-examples 12 --top-k 3 --chunk-size 1800 --chunk-overlap 100 --embedding-model model_cache_http\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embedding-cache vector_store\scifact_localpath_all-MiniLM_chunk1800.npz --model-cache-dir model_cache_http --hybrid-candidate-k 30 --rerank-candidate-k 30 --reranker-model model_cache_http\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8 --max-context-chars 2500 --max-output-tokens 300 --require-citations --output outputs\generation_eval_small_deepseek.jsonl --progress-every 1
```

## 9. 项目亮点与局限

项目亮点如下：

- 实现了完整 RAG 检索与生成链路，而非单纯调用 LLM API。
- 对比了 BM25、embedding、hybrid 和 rerank 多种检索策略。
- 使用 Hit@k、真正的 Recall@k、MRR 和 nDCG 做离线检索评估。
- 通过 chunk size、top-k、retriever 对比完成消融实验。
- 接入真实 LLM API，验证了端到端生成流程。
- 引入引用输出和证据不足拒答，体现事实一致性与幻觉控制意识。
- 通过错误分析解释了 rerank 效果不稳定的原因。

当前局限如下：

- 生成评估仍以人工复查为主，尚未引入自动 judge 或成规模人工标注。
- SciFact 是公开 benchmark，格式较规整，与真实企业知识库仍有差异。
- chunk 策略仍以固定长度切分为主，尚未加入标题层级、段落结构或语义切分。
- embedding 和 reranker 使用通用模型，未针对科学文献或生物医学语料做领域适配。
- query rewrite、多轮对话、tool use 和前端界面仍属于后续扩展方向。

## 10. 简历包装

本项目已经具备简历项目的基本完整度。原因如下：

- 有明确应用场景：面向本地/私有文档的 RAG 问答。
- 有完整技术链路：文档解析、chunk、BM25、embedding、hybrid retrieval、rerank、prompt、LLM 生成、引用和拒答。
- 有离线指标：Hit@k、真正的 Recall@k、MRR 和 nDCG。
- 有消融实验：对比 chunk size、top-k、retriever、rerank。
- 有错误分析：解释 hybrid 与 rerank 的差异和失败原因。
- 有端到端验证：接入 DeepSeek API 完成真实生成评估。

可用于简历的项目描述：

```text
基于 RAG 的本地知识库问答系统
- 构建面向本地文档的 RAG 问答系统，支持文档解析、chunk 切分、BM25 检索、embedding 语义检索、hybrid retrieval、cross-encoder rerank 和基于引用的答案生成。
- 基于 BEIR SciFact 构建离线检索评估流程，对比不同 chunk size、top-k、BM25、embedding、hybrid 和 rerank 对历史 Hit@k、MRR 的影响。
- 实验发现 hybrid retrieval 整体最稳定，在 chunk_size=1800、top_k=10 时达到 Hit@10=0.8333、MRR@10=0.6688；补充 rerank 后 Hit@10=0.8367、MRR@10=0.6588，说明 rerank 未在当前配置下稳定提升排序质量。
- 设计错误分析流程，对 hybrid 与 rerank 的命中差异进行分桶，分析关键词不重合、表面词误导、否定/比较关系困难和领域模型不适配等失败原因。
- 接入 DeepSeek API 完成小规模端到端生成评估，验证引用输出和证据不足拒答逻辑；12 条历史评估样例中 citation_format_valid_rate=0.9167、refusal_rate=0.4167（均非语义正确性指标）。
```

若简历篇幅较短，可压缩为三条：

```text
- 构建基于 RAG 的本地知识库问答系统，支持 BM25、embedding、hybrid retrieval、cross-encoder rerank、引用生成和证据不足拒答。
- 在 BEIR SciFact 上完成检索消融实验，对比 chunk size、top-k 与检索策略对历史 Hit@k、MRR 的影响；hybrid 在 chunk_size=1800、top_k=10 下达到 Hit@10=0.8333、MRR@10=0.6688。
- 接入 DeepSeek API 做端到端生成评估，并通过错误分析定位检索失败、rerank 排序不稳定和事实验证类问题中的幻觉风险。
```

## 11. 后续工作

后续优化方向包括：

- 生成评估：在已有 12 条真实 LLM 输出基础上，补充人工标注 `answer correctness`、`citation correctness` 和 `refusal correctness`。
- 错误分析扩展：将前 100 条扩展到完整 300 条，并分别比较 BM25 vs embedding、embedding vs hybrid、hybrid vs rerank。
- Reranker 优化：尝试科学文献或生物医学领域 reranker，并对比不同 `rerank_candidate_k`。
- Chunk 策略优化：加入按标题、段落或句子边界切分，并比较对 Recall、MRR、生成质量和 token 成本的影响。
- Query rewrite：对短 claim、术语不重合 query 和失败 case 做改写，观察召回是否改善。
- 展示层：增加轻量 Streamlit 界面，用于展示问题、答案、引用证据、检索分数和配置参数。

## 12. 总结

本项目已经形成一个完整的 RAG 应用算法实验闭环：从文档建库、召回、融合、重排，到离线评估、消融实验、错误分析和 LLM 生成验证。实验结果显示，hybrid retrieval 在 SciFact 上整体最稳定；cross-encoder rerank 在当前配置下没有稳定提升排序质量，说明 RAG 模块优化需要通过消融实验和错误分析验证。DeepSeek 端到端评估进一步说明，检索命中时模型能够生成带引用的回答，证据不足时具备一定拒答能力。后续工作的重点应放在人工生成评估、事实验证式输出格式、领域模型适配和更结构化的 chunk 策略上。
