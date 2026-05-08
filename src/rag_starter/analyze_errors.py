from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from .chunking import split_documents
from .eval_retrieval import build_retriever, load_eval_set
from .loaders import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two retrievers and export retrieval error cases.")
    parser.add_argument("--docs", default="data/scifact_docs")
    parser.add_argument("--eval-file", default="eval/scifact_eval.jsonl")
    parser.add_argument("--method-a", default="bm25")
    parser.add_argument("--method-b", default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-cache", default="vector_store/scifact_all-MiniLM_chunk1200.npz")
    parser.add_argument("--model-cache-dir", default="model_cache_http")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-base", choices=["bm25", "embedding", "faiss", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--output", default="outputs/error_analysis.jsonl")
    args = parser.parse_args()

    documents = load_documents(args.docs)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    examples = load_eval_set(args.eval_file)[: args.max_examples]

    retriever_a = build_retriever(make_args(args, args.method_a), chunks)
    retriever_b = build_retriever(make_args(args, args.method_b), chunks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"a_only": 0, "b_only": 0, "both_hit": 0, "both_miss": 0}
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            relevant_sources = set(example.get("relevant_sources", []))
            results_a = retriever_a.search(example["question"], top_k=args.top_k)
            results_b = retriever_b.search(example["question"], top_k=args.top_k)
            hit_a = hit(results_a, relevant_sources)
            hit_b = hit(results_b, relevant_sources)
            bucket = bucket_name(hit_a, hit_b)
            counts[bucket] += 1
            if bucket != "both_hit":
                file.write(
                    json.dumps(
                        {
                            "bucket": bucket,
                            "question": example["question"],
                            "relevant_sources": list(relevant_sources),
                            "method_a": args.method_a,
                            "method_b": args.method_b,
                            "top_a": summarize(results_a),
                            "top_b": summarize(results_b),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    total = len(examples)
    print(f"examples={total}")
    for key, value in counts.items():
        print(f"{key}={value}")
    print(f"output={output_path}")


def make_args(args: argparse.Namespace, retriever_name: str) -> Namespace:
    return Namespace(
        retriever=retriever_name,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache,
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


def hit(results: list, relevant_sources: set[str]) -> bool:
    return any(result.chunk.source in relevant_sources for result in results)


def bucket_name(hit_a: bool, hit_b: bool) -> str:
    if hit_a and hit_b:
        return "both_hit"
    if hit_a and not hit_b:
        return "a_only"
    if hit_b and not hit_a:
        return "b_only"
    return "both_miss"


def summarize(results: list) -> list[dict]:
    return [
        {
            "rank": result.rank,
            "score": result.score,
            "source": result.chunk.source,
            "preview": result.chunk.text.replace("\n", " ")[:220],
        }
        for result in results[:5]
    ]


if __name__ == "__main__":
    main()
