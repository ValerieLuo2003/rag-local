from __future__ import annotations

import argparse

from .chunking import split_documents
from .embedding_retrieval import EmbeddingRetriever
from .faiss_retrieval import FaissRetriever
from .generator import build_prompt, evidence_only_answer
from .hybrid_retrieval import HybridRetriever
from .loaders import load_documents
from .rerank_retrieval import RerankRetriever
from .retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RAG retrieval baseline.")
    parser.add_argument("--docs", default="data/sample_docs", help="File or directory containing docs.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size measured in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap measured in characters.")
    parser.add_argument("--retriever", choices=["bm25", "embedding", "faiss", "hybrid", "rerank"], default="bm25", help="Retrieval method.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used when --retriever embedding.",
    )
    parser.add_argument("--embedding-cache", help="Optional .npz cache path for chunk embeddings.")
    parser.add_argument("--model-cache-dir", default="model_cache", help="Directory for downloaded embedding models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
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
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt that will be sent to an LLM.")
    args = parser.parse_args()

    documents = load_documents(args.docs)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever = build_retriever(args, chunks)
    results = retriever.search(args.question, top_k=args.top_k)

    print(evidence_only_answer(args.question, results))
    if args.show_prompt:
        print("\n" + "=" * 80)
        print(build_prompt(args.question, results))


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


if __name__ == "__main__":
    main()
