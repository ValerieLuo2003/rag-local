from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

from .chunking import split_documents
from .eval_retrieval import build_retriever, load_eval_set, split_csv
from .experiment_tracking import (
    add_config_argument,
    build_run_metadata,
    parse_args_with_config,
    set_seed,
    write_json,
)
from .loaders import load_documents
from .retrieval_metrics import evaluate_retrieval_records


def main() -> None:
    parser = build_parser()
    args = parse_args_with_config(parser)
    set_seed(args.seed)

    retrievers = split_csv(args.retrievers)
    chunk_sizes = [int(value) for value in split_csv(args.chunk_sizes)]
    top_ks = [int(value) for value in split_csv(args.top_ks)]
    max_top_k = max(top_ks)

    documents = load_documents(args.docs)
    examples = load_eval_set(args.eval_file)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    maximum_chunk_count = 0

    for chunk_size in chunk_sizes:
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=args.chunk_overlap)
        maximum_chunk_count = max(maximum_chunk_count, len(chunks))
        for retriever_name in retrievers:
            run_args = make_retriever_args(args, retriever_name, chunk_size)
            retriever = build_retriever(run_args, chunks)
            records = []
            for index, example in enumerate(examples, start=1):
                results = retriever.search(example["question"], top_k=max_top_k)
                records.append(
                    {
                        **example,
                        "retrieved_sources": [result.chunk.source for result in results],
                    }
                )
                if args.progress and index % 50 == 0:
                    print(f"chunk={chunk_size} retriever={retriever_name} processed={index}/{len(examples)}")

            for top_k in top_ks:
                metrics = evaluate_retrieval_records(records, top_k=top_k)["overall"]
                row = {
                    "retriever": retriever_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "top_k": top_k,
                    "queries": metrics["queries"],
                    "judged_queries": metrics["judged_queries"],
                    "hit_at_k": metric_text(metrics["hit_at_k"]),
                    "recall_at_k": metric_text(metrics["recall_at_k"]),
                    "mrr_at_k": metric_text(metrics["mrr_at_k"]),
                    "ndcg_at_k": metric_text(metrics["ndcg_at_k"]),
                }
                rows.append(row)
                print(
                    f"{retriever_name}\tchunk={chunk_size}\ttop_k={top_k}\t"
                    f"Hit={row['hit_at_k']}\tRecall={row['recall_at_k']}\t"
                    f"MRR={row['mrr_at_k']}\tnDCG={row['ndcg_at_k']}"
                )

    if not rows:
        raise ValueError("No ablation rows were produced")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.metadata_output or str(output_path.with_suffix(".metadata.json"))
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "run": build_run_metadata(
                args,
                documents=len(documents),
                chunks=maximum_chunk_count,
                examples=len(examples),
                eval_file=args.eval_file,
            ),
            "output_csv": str(output_path),
            "rows": len(rows),
            "metric_note": "Metrics are macro-averaged at source level; duplicate chunks from one source count once.",
        },
    )
    print(f"output={output_path}")
    print(f"metadata_output={metadata_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chunk-size/top-k retrieval ablation experiments.")
    add_config_argument(parser)
    parser.add_argument("--docs", default="data/scifact_docs")
    parser.add_argument("--eval-file", default="eval/scifact_eval.jsonl")
    parser.add_argument("--retrievers", default="bm25,embedding,hybrid")
    parser.add_argument("--chunk-sizes", default="600,1200,1800")
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-ks", default="3,5,10")
    parser.add_argument("--max-examples", type=int, default=300, help="0 means all examples.")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-cache-dir", default="model_cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-base", choices=["bm25", "embedding", "faiss", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/ablation_results.csv")
    parser.add_argument("--metadata-output")
    parser.add_argument("--progress", action="store_true")
    return parser


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


def evaluate_results(results_by_query: list[tuple[dict, list]], top_k: int) -> dict:
    """Compatibility helper used by notebooks and older scripts."""

    records = [
        {
            **example,
            "retrieved_sources": [result.chunk.source for result in results],
        }
        for example, results in results_by_query
    ]
    return evaluate_retrieval_records(records, top_k=top_k)["overall"]


def metric_text(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
