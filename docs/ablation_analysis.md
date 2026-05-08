# Ablation Analysis: Chunk Size, Top-k, and Retriever

## 1. Experiment Setup

Dataset: BEIR SciFact

Evaluation set: `eval/scifact_eval.jsonl`

Number of queries: 300

Chunk overlap: 100 characters

Compared retrievers:

- BM25: keyword-based sparse retrieval.
- Embedding: dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
- Hybrid: RRF fusion of BM25 and embedding retrieval.

Compared chunk sizes:

- 600
- 1200
- 1800

Compared top-k values:

- 3
- 5
- 10

Metrics:

- Recall@k: whether the gold source document appears in the top-k retrieved chunks.
- MRR@k: how early the first gold source appears in the top-k results.

## 2. Full Results

| Retriever | Chunk Size | Top-k | Recall | MRR |
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

## 3. Main Conclusions

### Conclusion 1: Increasing top-k improves Recall, but costs more context

For every retriever and chunk size, Recall increases when top-k changes from 3 to 5 to 10.

Example with `chunk_size=1800`:

| Retriever | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.7100 | 0.7400 | 0.7967 |
| Embedding | 0.6800 | 0.7633 | 0.8133 |
| Hybrid | 0.7333 | 0.7700 | 0.8333 |

Interpretation:

- Larger top-k gives the retriever more chances to include the gold evidence.
- But in a real RAG system, larger top-k also means more prompt tokens, more noise, higher cost, and potentially worse generation.

Practical choice:

- Use `top_k=10` for retrieval evaluation.
- Use `top_k=3` or `top_k=5` for LLM answer generation to control token cost.

### Conclusion 2: Hybrid is the most stable retriever

Hybrid is generally the strongest method across chunk sizes and top-k values.

At `chunk_size=1800, top_k=10`:

| Retriever | Recall@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.7967 | 0.6378 |
| Embedding | 0.8133 | 0.6088 |
| Hybrid | 0.8333 | 0.6688 |

Interpretation:

- BM25 contributes exact lexical matching for scientific terms, numbers, and phrases.
- Embedding retrieval contributes semantic matching when the query and document use different wording.
- RRF fusion combines rankings without requiring BM25 scores and embedding cosine scores to be on the same scale.

Interview phrasing:

> BM25 and embedding retrieval have complementary strengths. In the SciFact experiment, hybrid retrieval improved Recall@10 and MRR@10 compared with either method alone, showing that lexical and semantic signals are both useful for scientific document retrieval.

### Conclusion 3: Larger chunks help BM25 and Hybrid in this dataset

For BM25, performance consistently improves as chunk size increases.

BM25 at `top_k=10`:

| Chunk Size | Recall@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7700 | 0.5953 |
| 1200 | 0.7733 | 0.6127 |
| 1800 | 0.7967 | 0.6378 |

Hybrid at `top_k=10`:

| Chunk Size | Recall@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.8333 | 0.6336 |
| 1200 | 0.8300 | 0.6414 |
| 1800 | 0.8333 | 0.6688 |

Interpretation:

- SciFact documents are short scientific abstracts.
- Larger chunks preserve more complete abstract-level context.
- For this dataset, chunk size 1800 often keeps the relevant evidence together instead of splitting it across chunks.

Caveat:

- This does not mean larger chunks are always better.
- For long enterprise documents, oversized chunks may introduce noise and increase LLM prompt cost.

### Conclusion 4: Embedding retrieval improves Recall but not always MRR

Embedding Recall@10 improves as chunk size grows:

| Chunk Size | Recall@10 | MRR@10 |
|---:|---:|---:|
| 600 | 0.7933 | 0.6130 |
| 1200 | 0.8067 | 0.6009 |
| 1800 | 0.8133 | 0.6088 |

Interpretation:

- Dense embedding retrieval is good at finding semantically related evidence.
- But the gold evidence is not always ranked at the very top.
- This explains why embedding improves Recall but MRR is not always better than BM25 or Hybrid.

This motivates reranking:

> Embedding retrieval is useful for recall, but reranking is needed to improve the final top-k order.

## 4. Best Configurations

### Best non-rerank configuration

Best overall non-rerank result:

```text
retriever=hybrid
chunk_size=1800
top_k=10
Recall@10=0.8333
MRR@10=0.6688
```

### Best known rerank configuration from previous experiment

Previous rerank experiment:

```text
retriever=rerank
rerank_base=hybrid
chunk_size=1200
top_k=10
Recall@10=0.8500
MRR@10=0.6594
```

Comparison:

- Rerank at chunk size 1200 improved Recall@10 to 0.8500.
- Hybrid at chunk size 1800 achieved a higher MRR@10 of 0.6688 without rerank.
- These are not fully comparable because rerank has not yet been tested at chunk size 1800.

Next recommended experiment:

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --retriever rerank --rerank-base hybrid --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1800 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1800.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --progress-every 50
```

## 5. Recommended Settings

For retrieval evaluation:

```text
retriever=hybrid or rerank
chunk_size=1800
top_k=10
chunk_overlap=100
```

For LLM generation with API cost control:

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

Reason:

- Retrieval evaluation can use larger top-k to measure recall.
- LLM generation should use smaller top-k to reduce token cost and avoid injecting too much irrelevant context.

## 6. Failure Analysis Direction

The earlier query:

```text
0-dimensional biomaterials show inductive properties.
```

failed because the retrieved evidence was not the gold source document. DeepSeek correctly refused to answer, which means the generation layer behaved safely. The real issue was retrieval failure.

This is a useful project finding:

> RAG answer quality is bounded by retrieval quality. If the correct evidence is not retrieved, a safe LLM should refuse rather than hallucinate.

Possible improvements:

- Query rewriting: rewrite scientific claims into better search queries.
- Domain-specific embedding model: use a biomedical/scientific sentence embedding model.
- Domain-specific reranker: use a reranker trained on scientific retrieval data.
- Multi-stage retrieval: use larger candidate sets before rerank, such as top-100 instead of top-50.
- Metadata-aware retrieval: preserve title, abstract, and document id fields separately.

## 7. Interview Summary

One-minute explanation:

> I ran ablation experiments on BEIR SciFact to compare BM25, dense embedding retrieval, and hybrid retrieval under different chunk sizes and top-k values. The results show that increasing top-k improves Recall, but increases context cost. Hybrid retrieval is the most stable method because it combines BM25's lexical matching and embedding retrieval's semantic matching. In this dataset, larger chunks such as 1800 characters work better because SciFact documents are short scientific abstracts, so larger chunks preserve complete evidence. The best non-rerank setup is hybrid retrieval with chunk size 1800 and top-k 10, reaching Recall@10 of 0.8333 and MRR@10 of 0.6688. This motivates further testing hybrid plus cross-encoder reranking at chunk size 1800.

