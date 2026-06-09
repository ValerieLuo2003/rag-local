from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer_generation import (
    AnswerResult,
    build_answer_generator,
    build_grounded_prompt,
    estimate_tokens,
    postprocess_answer,
    refusal_result,
    should_refuse,
)
from .answer_cli import add_llm_args, add_retrieval_args
from .chunking import split_documents
from .cli import build_retriever
from .eval_retrieval import load_eval_set
from .loaders import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded answer generation.")
    add_retrieval_args(parser, question_required=False)
    add_llm_args(parser)
    parser.add_argument("--eval-file", default="eval/eval_set.jsonl")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--output", default="outputs/generation_eval.jsonl")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    documents = load_documents(args.docs)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever = build_retriever(args, chunks)
    generator = None
    examples = load_eval_set(args.eval_file)[: args.max_examples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    evidence_hits = 0
    citation_valid = 0
    refused_count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, start=1):
            question = example["question"]
            evidence = retriever.search(question, top_k=args.top_k)
            relevant_sources = set(example.get("relevant_sources", []))
            retrieved_sources = [result.chunk.source for result in evidence]
            evidence_hit = any(source in relevant_sources for source in retrieved_sources)

            refused, reason = should_refuse(
                evidence,
                min_evidence=args.min_evidence,
                min_top_score=args.min_top_score,
            )
            prompt = build_grounded_prompt(question, evidence, max_context_chars=args.max_context_chars)
            estimated_prompt_tokens = estimate_tokens(prompt)
            estimated_total_token_budget = estimated_prompt_tokens + args.max_output_tokens

            if refused:
                result = refusal_result(args.llm_provider, args.llm_model, len(evidence), reason)
            elif args.dry_run:
                result = AnswerResult(
                    answer="DRY_RUN: LLM call skipped.",
                    provider="dry-run",
                    model=args.llm_model,
                    evidence_count=len(evidence),
                    refused=False,
                    citations_valid=True,
                )
            else:
                if generator is None:
                    generator = build_answer_generator(
                        provider=args.llm_provider,
                        model=args.llm_model,
                        api_key_env=args.api_key_env,
                        base_url=args.openai_base_url,
                        max_output_tokens=args.max_output_tokens,
                        max_context_chars=args.max_context_chars,
                        thinking=args.thinking,
                    )
                result = generator.generate(question, evidence)
                result = postprocess_answer(result, len(evidence), require_citations=args.require_citations)

            total += 1
            evidence_hits += int(evidence_hit)
            citation_valid += int(result.citations_valid)
            refused_count += int(result.refused)

            row = {
                "question": question,
                "relevant_sources": list(relevant_sources),
                "retrieved_sources": retrieved_sources,
                "evidence_hit": evidence_hit,
                "answer": result.answer,
                "provider": result.provider,
                "model": result.model,
                "refused": result.refused,
                "citations_valid": result.citations_valid,
                "citation_warning": result.citation_warning,
                "dry_run": args.dry_run,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_total_token_budget": estimated_total_token_budget,
            }
            if args.show_prompt:
                row["prompt"] = prompt
            if args.show_evidence:
                row["evidence"] = summarize_evidence(evidence)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"processed={index}/{len(examples)}")

    print(f"examples={total}")
    print(f"evidence_hit_rate={safe_div(evidence_hits, total):.4f}")
    print(f"citation_valid_rate={safe_div(citation_valid, total):.4f}")
    print(f"refusal_rate={safe_div(refused_count, total):.4f}")
    print(f"output={output_path}")
    print("answer_correctness requires human labels or a judge model; this script records answers for review.")


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_evidence(evidence: list) -> list[dict]:
    return [
        {
            "rank": result.rank,
            "score": result.score,
            "source": result.chunk.source,
            "chunk_id": result.chunk.chunk_id,
            "preview": result.chunk.text.replace("\n", " ")[:360],
        }
        for result in evidence
    ]


if __name__ == "__main__":
    main()
