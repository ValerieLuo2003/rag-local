from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .answer_generation import (
    AnswerResult,
    build_answer_generator,
    build_grounded_prompt,
    estimate_tokens,
    extract_citation_indices,
    postprocess_answer,
    refusal_result,
    should_refuse,
)
from .answer_cli import add_llm_args, add_retrieval_args
from .chunking import split_documents
from .cli import build_retriever
from .eval_retrieval import load_eval_set
from .experiment_tracking import (
    add_config_argument,
    build_run_metadata,
    parse_args_with_config,
    set_seed,
    write_json,
)
from .generation_metrics import aggregate_generation_records
from .loaders import load_documents
from .score_generation import print_summary


def main() -> None:
    parser = build_parser()
    args = parse_args_with_config(parser)
    set_seed(args.seed)

    documents = load_documents(args.docs)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever = build_retriever(args, chunks)
    generator = None
    examples = load_eval_set(args.eval_file)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if len(examples) < args.min_examples:
        raise ValueError(
            f"Evaluation set has {len(examples)} examples; --min-examples requires {args.min_examples}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with output_path.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, start=1):
            total_started_at = time.perf_counter()
            question = example["question"]

            retrieval_started_at = time.perf_counter()
            evidence = retriever.search(question, top_k=args.top_k)
            retrieval_latency_ms = (time.perf_counter() - retrieval_started_at) * 1000

            relevant_sources = {str(source) for source in example.get("relevant_sources", [])}
            retrieved_sources = [result.chunk.source for result in evidence]
            evidence_hit = bool(relevant_sources.intersection(retrieved_sources))
            answerable, answerable_inferred = resolve_answerable(example, relevant_sources)

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
                    citations_valid=False,
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

            citation_proxies = citation_source_proxies(
                result.answer,
                evidence,
                relevant_sources,
                refused=result.refused,
            )
            total_latency_ms = (time.perf_counter() - total_started_at) * 1000
            estimated_cost_usd = calculate_cost(
                result.input_tokens,
                result.output_tokens,
                args.input_cost_per_1m,
                args.output_cost_per_1m,
            )

            row = {
                "query_id": str(example.get("query_id", index)),
                "question": question,
                "query_type": example.get("query_type", example.get("type", "unknown")),
                "review_group": example.get("review_group", "unknown"),
                "answerable": answerable,
                "answerable_inferred": answerable_inferred,
                "reference_answer": example.get("reference_answer"),
                "relevant_sources": sorted(relevant_sources),
                "retrieved_sources": retrieved_sources,
                "evidence_hit": evidence_hit,
                "answer": result.answer,
                "provider": result.provider,
                "model": result.model,
                "refused": result.refused,
                "refusal_reason": reason if refused else "",
                "citations_valid": result.citations_valid,
                "citation_warning": result.citation_warning,
                **citation_proxies,
                "dry_run": args.dry_run,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_total_token_budget": estimated_total_token_budget,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": result.latency_ms,
                "total_latency_ms": total_latency_ms,
                "evaluation": example.get("evaluation", {}),
            }
            if args.show_prompt:
                row["prompt"] = prompt
            if args.show_evidence:
                row["evidence"] = summarize_evidence(evidence)
            records.append(row)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"processed={index}/{len(examples)}")

    metrics = aggregate_generation_records(records, group_by=args.group_by or None)
    summary_path = args.summary_output or str(output_path.with_suffix(".summary.json"))
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "run": build_run_metadata(
                args,
                documents=len(documents),
                chunks=len(chunks),
                examples=len(examples),
                eval_file=args.eval_file,
            ),
            "metrics": metrics,
            "metric_note": {
                "citations_valid": "Syntax/range check only.",
                "citation_source_precision_proxy": "Whether cited chunks come from gold sources; not semantic entailment.",
                "human_or_judge_scores": "Must be filled in evaluation fields after answer review.",
            },
        },
    )

    print_summary(metrics["overall"])
    print(f"output={output_path}")
    print(f"summary_output={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate grounded answer generation.")
    add_config_argument(parser)
    add_retrieval_args(parser, question_required=False)
    add_llm_args(parser)
    parser.add_argument("--eval-file", default="eval/eval_set.jsonl")
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all examples.")
    parser.add_argument(
        "--min-examples",
        type=int,
        default=0,
        help="Fail if fewer examples are available. Use 100 for an official project report.",
    )
    parser.add_argument("--group-by", default="query_type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-cost-per-1m", type=float, help="USD per one million input tokens.")
    parser.add_argument("--output-cost-per-1m", type=float, help="USD per one million output tokens.")
    parser.add_argument("--output", default="outputs/generation_eval.jsonl")
    parser.add_argument("--summary-output")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def resolve_answerable(example: dict, relevant_sources: set[str]) -> tuple[bool, bool]:
    value = example.get("answerable")
    if isinstance(value, bool):
        return value, False
    return bool(relevant_sources), True


def citation_source_proxies(answer: str, evidence: list, relevant_sources: set[str], *, refused: bool) -> dict:
    if refused:
        return {
            "cited_evidence_indices": [],
            "cited_sources": [],
            "citation_source_precision": None,
            "citation_source_recall": None,
        }

    valid_indices = sorted(
        {
            index
            for index in extract_citation_indices(answer)
            if 1 <= index <= len(evidence)
        }
    )
    cited_sources = {evidence[index - 1].chunk.source for index in valid_indices}
    source_precision = None
    if cited_sources:
        source_precision = len(cited_sources.intersection(relevant_sources)) / len(cited_sources)
    source_recall = None
    if relevant_sources:
        source_recall = len(cited_sources.intersection(relevant_sources)) / len(relevant_sources)
    return {
        "cited_evidence_indices": valid_indices,
        "cited_sources": sorted(cited_sources),
        "citation_source_precision": source_precision,
        "citation_source_recall": source_recall,
    }


def calculate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if input_cost_per_1m is None and output_cost_per_1m is None:
        return None
    input_rate = input_cost_per_1m or 0.0
    output_rate = output_cost_per_1m or 0.0
    return input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000


def summarize_evidence(evidence: list) -> list[dict]:
    return [
        {
            "rank": result.rank,
            "score": result.score,
            "source": result.chunk.source,
            "doc_id": result.chunk.doc_id,
            "title": result.chunk.title,
            "chunk_id": result.chunk.chunk_id,
            "section_id": result.chunk.section_id,
            "section_title": result.chunk.section_title,
            "page_start": result.chunk.page_start,
            "page_end": result.chunk.page_end,
            "canonical_url": result.chunk.canonical_url,
            "citation": result.chunk.citation_label(),
            "preview": result.chunk.text.replace("\n", " ")[:360],
        }
        for result in evidence
    ]


if __name__ == "__main__":
    main()
