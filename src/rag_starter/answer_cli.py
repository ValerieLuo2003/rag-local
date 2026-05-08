from __future__ import annotations

import argparse

from .answer_generation import (
    DEFAULT_LLM_MODEL,
    build_answer_generator,
    build_grounded_prompt,
    estimate_tokens,
    format_evidence,
    postprocess_answer,
    refusal_result,
    should_refuse,
)
from .chunking import split_documents
from .cli import build_retriever
from .loaders import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval plus grounded LLM answer generation.")
    add_retrieval_args(parser)
    add_llm_args(parser)
    args = parser.parse_args()

    log(args, "loading documents")
    documents = load_documents(args.docs)
    log(args, f"loaded documents: {len(documents)}")
    log(args, "splitting chunks")
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    log(args, f"built chunks: {len(chunks)}")
    log(args, f"building retriever: {args.retriever}")
    retriever = build_retriever(args, chunks)
    log(args, f"searching top-{args.top_k} evidence")
    evidence = retriever.search(args.question, top_k=args.top_k)
    log(args, f"retrieved evidence: {len(evidence)}")

    refused, reason = should_refuse(
        evidence,
        min_evidence=args.min_evidence,
        min_top_score=args.min_top_score,
    )
    prompt = build_grounded_prompt(args.question, evidence, max_context_chars=args.max_context_chars)
    if args.dry_run:
        print_dry_run(args, evidence, prompt, refused, reason)
        return

    if refused:
        result = refusal_result(
            provider=args.llm_provider,
            model=args.llm_model,
            evidence_count=len(evidence),
            reason=reason,
        )
    else:
        log(args, f"building answer generator: {args.llm_provider}")
        generator = build_answer_generator(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key_env=args.api_key_env,
            base_url=args.openai_base_url,
            max_output_tokens=args.max_output_tokens,
            max_context_chars=args.max_context_chars,
            thinking=args.thinking,
        )
        log(args, "generating answer")
        result = generator.generate(args.question, evidence)
        result = postprocess_answer(
            result,
            evidence_count=len(evidence),
            require_citations=args.require_citations,
        )

    print(f"provider={result.provider}")
    print(f"model={result.model}")
    print(f"evidence_count={result.evidence_count}")
    print(f"refused={result.refused}")
    print(f"citations_valid={result.citations_valid}")
    if result.citation_warning:
        print(f"citation_warning={result.citation_warning}")
    print("\nAnswer:")
    print(result.answer)

    if args.show_evidence:
        print("\n" + "=" * 80)
        print("Evidence:")
        print(format_evidence(evidence, max_context_chars=args.max_context_chars))


def add_retrieval_args(parser: argparse.ArgumentParser, question_required: bool = True) -> None:
    parser.add_argument("--docs", default="data/sample_docs", help="File or directory containing docs.")
    parser.add_argument("--question", required=question_required, help="User question.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final evidence chunks.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size measured in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap measured in characters.")
    parser.add_argument("--retriever", choices=["bm25", "embedding", "faiss", "hybrid", "rerank"], default="rerank")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-cache", help="Optional .npz cache path for chunk embeddings.")
    parser.add_argument("--model-cache-dir", default="model_cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-base", choices=["bm25", "embedding", "faiss", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", choices=["mock", "openai", "openai-chat"], default="mock")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--openai-base-url", help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--thinking", choices=["default", "enabled", "disabled"], default="default")
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--min-top-score", type=float)
    parser.add_argument("--require-citations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print prompt/token estimate without calling an LLM.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the exact prompt sent to the LLM.")
    parser.add_argument("--show-evidence", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")


def log(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(f"[answer_cli] {message}", flush=True)


def print_dry_run(
    args: argparse.Namespace,
    evidence,
    prompt: str,
    refused: bool,
    refusal_reason: str,
) -> None:
    prompt_tokens = estimate_tokens(prompt)
    print("dry_run=True")
    print(f"provider={args.llm_provider}")
    print(f"model={args.llm_model}")
    print(f"evidence_count={len(evidence)}")
    print(f"max_context_chars={args.max_context_chars}")
    print(f"prompt_chars={len(prompt)}")
    print(f"estimated_prompt_tokens={prompt_tokens}")
    print(f"max_output_tokens={args.max_output_tokens}")
    print(f"estimated_total_token_budget={prompt_tokens + args.max_output_tokens}")
    print(f"would_refuse={refused}")
    if refusal_reason:
        print(f"refusal_reason={refusal_reason}")
    if evidence:
        print("\nTop evidence sources:")
        for result in evidence:
            print(f"[{result.rank}] score={result.score:.4f} source={result.chunk.source}")
    if args.show_prompt:
        print("\n" + "=" * 80)
        print("Prompt:")
        print(prompt)


if __name__ == "__main__":
    main()
