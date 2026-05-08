from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import split_documents
from .embedding_retrieval import EmbeddingRetriever
from .faiss_retrieval import FaissRetriever
from .hybrid_retrieval import HybridRetriever
from .loaders import load_documents
from .rerank_retrieval import RerankRetriever
from .retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with source-level labels.")
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--eval-file", default="eval/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--retriever", choices=["bm25", "embedding", "faiss", "hybrid", "rerank"], default="bm25")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used when --retriever embedding.",
    )
    parser.add_argument("--embedding-cache", help="Optional .npz cache path for chunk embeddings.")
    parser.add_argument("--model-cache-dir", default="model_cache", help="Directory for downloaded embedding models.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50, help="Candidates from each retriever for hybrid fusion.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant used by hybrid retrieval.")
    parser.add_argument("--rerank-base", choices=["bm25", "embedding", "faiss", "hybrid"], default="hybrid", help="Base retriever used before rerank.")
    parser.add_argument("--rerank-candidate-k", type=int, default=50, help="Number of candidates passed to the reranker.")
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="CrossEncoder model used when --retriever rerank.",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=16, help="Reranker batch size.")
    parser.add_argument("--reranker-max-length", type=int, default=512, help="Reranker max sequence length.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N examples.")
    args = parser.parse_args()

    documents = load_documents(args.docs)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever = build_retriever(args, chunks)
    examples = load_eval_set(args.eval_file)

    hit_count = 0
    reciprocal_ranks = []

    for index, example in enumerate(examples, start=1):
        results = retriever.search(example["question"], top_k=args.top_k)
        retrieved_sources = [result.chunk.source for result in results]
        relevant_sources = set(example["relevant_sources"])
        first_rank = None
        for rank, source in enumerate(retrieved_sources, start=1):
            if source in relevant_sources:
                first_rank = rank
                break
        if first_rank is not None:
            hit_count += 1
            reciprocal_ranks.append(1 / first_rank)
        else:
            reciprocal_ranks.append(0.0)
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"processed={index}/{len(examples)}")

    total = len(examples)
    recall_at_k = hit_count / total if total else 0.0
    mrr = sum(reciprocal_ranks) / total if total else 0.0

    print(f"retriever={args.retriever}")
    if args.retriever in {"embedding", "faiss", "hybrid", "rerank"}:
        print(f"embedding_model={args.embedding_model}")
    if args.retriever == "hybrid" or (args.retriever == "rerank" and args.rerank_base == "hybrid"):
        print(f"hybrid_candidate_k={args.hybrid_candidate_k}")
        print(f"rrf_k={args.rrf_k}")
    if args.retriever == "rerank":
        print(f"rerank_base={args.rerank_base}")
        print(f"rerank_candidate_k={args.rerank_candidate_k}")
        print(f"reranker_model={args.reranker_model}")
    print(f"examples={total}")
    print(f"Recall@{args.top_k}={recall_at_k:.4f}")
    print(f"MRR@{args.top_k}={mrr:.4f}")


def build_retriever(args: argparse.Namespace, chunks):
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
    examples = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


if __name__ == "__main__":
    main()
