from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

from .chunking import split_documents
from .eval_retrieval import build_retriever, load_eval_set
from .loaders import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunk-size/top-k retrieval ablation experiments.")
    parser.add_argument("--docs", default="data/scifact_docs")
    parser.add_argument("--eval-file", default="eval/scifact_eval.jsonl")
    parser.add_argument("--retrievers", default="bm25,embedding,hybrid")
    parser.add_argument("--chunk-sizes", default="600,1200,1800")
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-ks", default="3,5,10")
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-cache-dir", default="model_cache_http")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-base", choices=["bm25", "embedding", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--output", default="outputs/ablation_results.csv")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    retrievers = split_csv(args.retrievers)
    chunk_sizes = [int(value) for value in split_csv(args.chunk_sizes)]
    top_ks = [int(value) for value in split_csv(args.top_ks)]
    max_top_k = max(top_ks)

    documents = load_documents(args.docs)
    examples = load_eval_set(args.eval_file)[: args.max_examples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for chunk_size in chunk_sizes:
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=args.chunk_overlap)
        for retriever_name in retrievers:
            run_args = make_retriever_args(args, retriever_name, chunk_size)
            retriever = build_retriever(run_args, chunks)
            results_by_query = []
            for index, example in enumerate(examples, start=1):
                results = retriever.search(example["question"], top_k=max_top_k)
                results_by_query.append((example, results))
                if args.progress and index % 50 == 0:
                    print(f"chunk={chunk_size} retriever={retriever_name} processed={index}/{len(examples)}")

            for top_k in top_ks:
                recall, mrr = evaluate_results(results_by_query, top_k)
                row = {
                    "retriever": retriever_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "top_k": top_k,
                    "examples": len(examples),
                    "recall": f"{recall:.4f}",
                    "mrr": f"{mrr:.4f}",
                }
                rows.append(row)
                print(
                    f"{retriever_name}\tchunk={chunk_size}\ttop_k={top_k}\t"
                    f"Recall={recall:.4f}\tMRR={mrr:.4f}"
                )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={output_path}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_retriever_args(args: argparse.Namespace, retriever_name: str, chunk_size: int) -> Namespace:
    embedding_cache = None
    if retriever_name in {"embedding", "faiss", "hybrid", "rerank"}:
        embedding_cache = f"vector_store/ablation_{retriever_name}_chunk{chunk_size}.npz"
    return Namespace(
        retriever=retriever_name,
        embedding_model=args.embedding_model,
        embedding_cache=embedding_cache,
        model_cache_dir=args.model_cache_dir,
        batch_size=args.batch_size,
        hybrid_candidate_k=args.hybrid_candidate_k,
        rrf_k=args.rrf_k,
        rerank_base=args.rerank_base,
        rerank_candidate_k=args.rerank_candidate_k,
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
    )


def evaluate_results(results_by_query: list[tuple[dict, list]], top_k: int) -> tuple[float, float]:
    hits = 0
    reciprocal_ranks = []
    for example, results in results_by_query:
        relevant_sources = set(example.get("relevant_sources", []))
        first_rank = None
        for rank, result in enumerate(results[:top_k], start=1):
            if result.chunk.source in relevant_sources:
                first_rank = rank
                break
        if first_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            hits += 1
            reciprocal_ranks.append(1 / first_rank)
    total = len(results_by_query)
    return hits / total if total else 0.0, sum(reciprocal_ranks) / total if total else 0.0


if __name__ == "__main__":
    main()
