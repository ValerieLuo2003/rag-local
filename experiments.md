# Retrieval Experiments

Dataset: BEIR SciFact converted by `rag_starter.import_scifact`.

Evaluation file: `eval/scifact_eval.jsonl`

Docs: `data/scifact_docs`

Chunking:

- `chunk_size=1200`
- `chunk_overlap=100`
- `top_k=10`
- `examples=300`

## Historical Results and Metric Correction

The old implementation incremented the metric when **any** relevant source appeared
in Top-K. Therefore the column previously called `Recall@10` was actually
source-level `Hit@10`. The current evaluator reports Hit, true macro Recall, MRR,
and nDCG separately and deduplicates multiple chunks from one source.

| Method | Hit@10 | Recall@10 | MRR@10 | nDCG@10 | Notes |
|---|---:|---:|---:|---:|---|
| BM25 | 0.7733 | 0.7519 | 0.6127 | 0.6398 | Recomputed with the strict evaluator. |
| Embedding | 0.8067 | pending | 0.6009 | pending | Hit/MRR copied from the historical run. |
| Hybrid | 0.8300 | pending | 0.6414 | pending | Hit/MRR copied from the historical run. |
| Hybrid + Rerank | 0.8500 | pending | 0.6594 | pending | Hit/MRR copied from the historical run. |

## Current Reading

Embedding retrieval improves historical Hit@10 over BM25, which means it finds at
least one gold source document for more queries. Its MRR@10 is slightly lower,
which suggests that the first correct document is not always ranked as high as
BM25 when both find it.

Hybrid retrieval improves both historical Hit@10 and MRR@10. In this setting,
BM25 contributes exact lexical matching while embedding retrieval contributes
semantic matching. RRF combines the two rankings without requiring their raw
scores to be on the same scale.

Rerank further improves both metrics. The base hybrid retriever aims to cover relevant candidates in the top-50 set, and the cross-encoder reranker then scores each `(query, chunk)` pair with deeper interaction before selecting the final top-10.

This motivates the next step:

1. Generation evaluation: measure answer correctness, citation hit rate, refusal accuracy, and hallucination rate.

Detailed chunk-size/top-k ablation analysis: `docs/ablation_analysis.md`

Chinese review note: `docs/ablation_analysis_zh.md`

## Cryptography Domain Development Run

The current portfolio-facing experiment uses 44 official NIST/RFC documents,
6,073 structured chunks, and 100 verified development questions.

| Method | Hit@10 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.9667 | 0.9611 | 0.8588 | 0.8821 | 5.71 ms |
| Agent + BM25 | 1.0000 | 1.0000 | 0.9778 | 0.9802 | 23.22 ms |

Configs:

- `configs/retrieval_crypto_100_bm25.json`
- `configs/retrieval_crypto_100_agent.json`
- `configs/generation_crypto_100_mock.json`

The Agent result was obtained after inspecting failures on this same 100-question
development set. It is not an unseen-test result. Full routing/error analysis and
the rejected obsolete-document demotion ablation are recorded in
`docs/crypto_eval_report.md`.
