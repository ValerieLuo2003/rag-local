from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .chunking import split_documents
from .agent_retrieval import AgenticRetriever
from .embedding_retrieval import EmbeddingRetriever
from .experiment_tracking import (
    add_config_argument,
    build_run_metadata,
    latency_summary,
    parse_args_with_config,
    set_seed,
    write_json,
)
from .faiss_retrieval import FaissRetriever
from .hybrid_retrieval import HybridRetriever
from .loaders import load_documents
from .rerank_retrieval import RerankRetriever
from .retrieval import BM25Retriever
from .retrieval_metrics import evaluate_retrieval_records, query_metrics_as_dict


def main() -> None:
    parser = build_parser()
    args = parse_args_with_config(parser)
    set_seed(args.seed)

    documents = load_documents(args.docs)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever = build_retriever(args, chunks)
    examples = load_eval_set(args.eval_file)
    group_by = split_csv(args.group_by)

    records = []
    latencies_ms = []
    for index, example in enumerate(examples, start=1):
        started_at = time.perf_counter()
        results = retriever.search(example["question"], top_k=args.top_k)
        latencies_ms.append((time.perf_counter() - started_at) * 1000)
        record = {
            **example,
            "retrieved_sources": [result.chunk.source for result in results],
            "retrieved": [
                {
                    "rank": result.rank,
                    "source": result.chunk.source,
                    "chunk_id": result.chunk.chunk_id,
                    "section_id": result.chunk.section_id,
                    "score": result.score,
                }
                for result in results
            ],
        }
        if hasattr(retriever, "last_plan"):
            record["agent_plan"] = retriever.last_plan
        records.append(record)
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"processed={index}/{len(examples)}")

    metrics = evaluate_retrieval_records(records, top_k=args.top_k, group_by=group_by)
    query_details = []
    for record in records:
        relevant_sources = record.get("relevant_sources", [])
        per_query_metrics = (
            query_metrics_as_dict(
                record["retrieved_sources"],
                relevant_sources,
                args.top_k,
            )
            if relevant_sources
            else None
        )
        query_details.append(
            {
                "query_id": record.get("query_id"),
                "question": record["question"],
                "query_type": record.get("query_type"),
                "relevant_sources": relevant_sources,
                "retrieved": record["retrieved"],
                "metrics": per_query_metrics,
                "agent_plan": record.get("agent_plan"),
            }
        )
    report = {
        "schema_version": 1,
        "run": build_run_metadata(
            args,
            documents=len(documents),
            chunks=len(chunks),
            examples=len(examples),
            eval_file=args.eval_file,
        ),
        "metrics": metrics,
        "query_details": query_details,
        "retrieval_latency_ms": latency_summary(latencies_ms),
    }
    output_path = write_json(args.output, report)

    print_report(args, report)
    print(f"output={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate source-level retrieval with strict IR metrics.")
    add_config_argument(parser)
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--eval-file", default="eval/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--retriever", choices=["bm25", "embedding", "faiss", "hybrid", "rerank", "agent"], default="bm25")
    parser.add_argument(
        "--agent-base",
        choices=["bm25", "embedding", "faiss", "hybrid"],
        default="bm25",
    )
    parser.add_argument("--agent-candidate-k", type=int, default=40)
    parser.add_argument("--agent-rrf-k", type=int, default=60)
    parser.add_argument("--agent-max-per-source", type=int, default=2)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used by dense retrieval.",
    )
    parser.add_argument("--embedding-cache", help="Optional .npz cache path for chunk embeddings.")
    parser.add_argument("--model-cache-dir", default="model_cache", help="Directory for downloaded models.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-base", choices=["bm25", "embedding", "faiss", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--group-by", default="type,review_group", help="Comma-separated query metadata fields.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/retrieval_eval.json")
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def print_report(args: argparse.Namespace, report: dict) -> None:
    overall = report["metrics"]["overall"]
    print(f"retriever={args.retriever}")
    print(f"examples={overall['queries']}")
    print(f"judged_queries={overall['judged_queries']}")
    print(f"Hit@{args.top_k}={format_metric(overall['hit_at_k'])}")
    print(f"Recall@{args.top_k}={format_metric(overall['recall_at_k'])}")
    print(f"MRR@{args.top_k}={format_metric(overall['mrr_at_k'])}")
    print(f"nDCG@{args.top_k}={format_metric(overall['ndcg_at_k'])}")
    print(f"mean_retrieval_latency_ms={format_metric(report['retrieval_latency_ms']['mean'])}")

    for field, groups in report["metrics"]["groups"].items():
        for label, values in groups.items():
            print(
                f"group[{field}={label}] queries={values['queries']} "
                f"Hit={format_metric(values['hit_at_k'])} "
                f"Recall={format_metric(values['recall_at_k'])} "
                f"MRR={format_metric(values['mrr_at_k'])} "
                f"nDCG={format_metric(values['ndcg_at_k'])}"
            )


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def build_retriever(args: argparse.Namespace, chunks):
    if args.retriever == "agent":
        base_retriever = build_base_retriever(args, chunks, args.agent_base)
        return AgenticRetriever(
            chunks,
            base_retriever,
            candidate_k=args.agent_candidate_k,
            rrf_k=args.agent_rrf_k,
            max_per_source=args.agent_max_per_source,
        )
    if args.retriever == "rerank":
        base_retriever = build_base_retriever(args, chunks, args.rerank_base)
        return RerankRetriever(
            base_retriever,
            candidate_k=args.rerank_candidate_k,
            model_name=args.reranker_model,
            batch_size=args.reranker_batch_size,
            model_cache_dir=args.model_cache_dir,
            max_length=args.reranker_max_length,
        )
    return build_base_retriever(args, chunks, args.retriever)


def build_base_retriever(args: argparse.Namespace, chunks, retriever_name: str):
    if retriever_name == "bm25":
        return BM25Retriever(chunks)
    if retriever_name == "embedding":
        return EmbeddingRetriever(
            chunks,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            cache_path=args.embedding_cache,
            model_cache_dir=args.model_cache_dir,
        )
    if retriever_name == "faiss":
        return FaissRetriever(
            chunks,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            cache_path=args.embedding_cache,
            model_cache_dir=args.model_cache_dir,
        )
    if retriever_name == "hybrid":
        return HybridRetriever(
            chunks,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            cache_path=args.embedding_cache,
            model_cache_dir=args.model_cache_dir,
            candidate_k=args.hybrid_candidate_k,
            rrf_k=args.rrf_k,
        )
    raise ValueError(f"Unknown retriever: {retriever_name}")


def load_eval_set(path: str | Path) -> list[dict]:
    eval_path = Path(path)
    if eval_path.suffix.lower() == ".json":
        with eval_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("parts"), list):
            raise ValueError(f"Evaluation manifest must contain a parts list: {eval_path}")
        examples = []
        for part in manifest["parts"]:
            part_path = Path(str(part))
            if not part_path.is_absolute():
                part_path = eval_path.parent / part_path
            examples.extend(load_eval_set(part_path))
        validate_unique_query_ids(examples, eval_path)
        expected = manifest.get("expected_examples")
        if expected is not None and len(examples) != int(expected):
            raise ValueError(
                f"Evaluation manifest expected {expected} examples, found {len(examples)}"
            )
        return examples

    examples = []
    seen_ids: set[str] = set()
    with eval_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if not example.get("question"):
                raise ValueError(f"Missing question at {path}:{line_number}")
            query_id = example.get("query_id")
            if query_id is not None:
                query_id = str(query_id)
                if query_id in seen_ids:
                    raise ValueError(f"Duplicate query_id={query_id!r} at {path}:{line_number}")
                seen_ids.add(query_id)
            examples.append(example)
    return examples


def validate_unique_query_ids(examples: list[dict], path: str | Path) -> None:
    seen_ids = set()
    for index, example in enumerate(examples, start=1):
        query_id = example.get("query_id")
        if query_id is None:
            continue
        query_id = str(query_id)
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id={query_id!r} in {path} at example {index}")
        seen_ids.add(query_id)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
