# Retrieval Experiments

Dataset: BEIR SciFact converted by `rag_starter.import_scifact`.

Evaluation file: `eval/scifact_eval.jsonl`

Docs: `data/scifact_docs`

Chunking:

- `chunk_size=1200`
- `chunk_overlap=100`
- `top_k=10`
- `examples=300`

## Results

| Method | Recall@10 | MRR@10 | Notes |
|---|---:|---:|---|
| BM25 | 0.7733 | 0.6127 | Keyword-based lexical retrieval baseline. |
| Embedding | 0.8067 | 0.6009 | `sentence-transformers/all-MiniLM-L6-v2`, exact NumPy cosine search. |
| Hybrid | 0.8300 | 0.6414 | RRF fusion over BM25 and embedding rankings, `hybrid_candidate_k=50`. |
| Hybrid + Rerank | 0.8500 | 0.6594 | Hybrid top-50 candidates reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`. |

## Current Reading

Embedding retrieval improves Recall@10 over BM25, which means it finds the gold source document for more queries. Its MRR@10 is slightly lower, which suggests that the correct document is not always ranked as high as BM25 when both find it.

Hybrid retrieval improves both Recall@10 and MRR@10. In this setting, BM25 contributes exact lexical matching while embedding retrieval contributes semantic matching. RRF combines the two rankings without requiring their raw scores to be on the same scale.

Rerank further improves both metrics. The base hybrid retriever aims to cover relevant candidates in the top-50 set, and the cross-encoder reranker then scores each `(query, chunk)` pair with deeper interaction before selecting the final top-10.

This motivates the next step:

1. Generation evaluation: measure answer correctness, citation hit rate, refusal accuracy, and hallucination rate.

Detailed chunk-size/top-k ablation analysis: `docs/ablation_analysis.md`

Chinese review note: `docs/ablation_analysis_zh.md`
