from __future__ import annotations

import argparse
from collections import Counter

from .eval_retrieval import load_eval_set
from .loaders import load_document_file, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generation/retrieval evaluation JSONL.")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--require-reference-answer", action="store_true")
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--manifest", help="Validate gold source ids against this corpus manifest.")
    parser.add_argument(
        "--validate-sections",
        action="store_true",
        help="Parse referenced documents and verify gold section ids (requires --manifest).",
    )
    args = parser.parse_args()

    examples = load_eval_set(args.eval_file)
    errors = validate_examples(
        examples,
        min_examples=args.min_examples,
        require_reference_answer=args.require_reference_answer,
        require_verified=args.require_verified,
    )
    if args.validate_sections and not args.manifest:
        errors.append("--validate-sections requires --manifest")
    if args.manifest:
        errors.extend(
            validate_corpus_references(
                examples,
                args.manifest,
                validate_sections=args.validate_sections,
            )
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    query_types = Counter(example["query_type"] for example in examples)
    answerability = Counter("answerable" if example["answerable"] else "unanswerable" for example in examples)
    print(f"examples={len(examples)}")
    print(f"query_types={dict(sorted(query_types.items()))}")
    print(f"answerability={dict(sorted(answerability.items()))}")
    print("validation=passed")


def validate_examples(
    examples: list[dict],
    *,
    min_examples: int = 100,
    require_reference_answer: bool = False,
    require_verified: bool = False,
) -> list[str]:
    errors = []
    if len(examples) < min_examples:
        errors.append(f"expected at least {min_examples} examples, found {len(examples)}")

    for index, example in enumerate(examples, start=1):
        label = f"example {index}"
        if example.get("query_id") in (None, ""):
            errors.append(f"{label}: query_id is required")
        if not isinstance(example.get("query_type"), str) or not example["query_type"].strip():
            errors.append(f"{label}: query_type is required")
        if not isinstance(example.get("answerable"), bool):
            errors.append(f"{label}: answerable must be boolean")
        relevant_sources = example.get("relevant_sources")
        if not isinstance(relevant_sources, list):
            errors.append(f"{label}: relevant_sources must be a list")
        elif example.get("answerable") is True and not relevant_sources:
            errors.append(f"{label}: answerable query must have at least one relevant source")
        if require_reference_answer and not str(example.get("reference_answer", "")).strip():
            errors.append(f"{label}: reference_answer is required")
        if require_verified:
            if example.get("review_status") != "verified":
                errors.append(f"{label}: review_status must be 'verified'")
            if not str(example.get("reviewer", "")).strip():
                errors.append(f"{label}: reviewer is required")
    return errors


def validate_corpus_references(
    examples: list[dict],
    manifest_path: str,
    *,
    validate_sections: bool = False,
) -> list[str]:
    entries = {entry["doc_id"]: entry for entry in load_manifest(manifest_path)}
    errors = []
    referenced_sources = {
        str(source)
        for example in examples
        for source in example.get("relevant_sources", [])
    }
    for source in sorted(referenced_sources - set(entries)):
        errors.append(f"unknown relevant source: {source}")
    if not validate_sections:
        return errors

    section_catalog = {}
    for source in sorted(referenced_sources & set(entries)):
        document = load_document_file(entries[source]["local_path"], metadata=entries[source])
        section_catalog[source] = {section.section_id for section in document.sections}

    for index, example in enumerate(examples, start=1):
        label = f"example {index}"
        relevant_sources = {
            str(source)
            for source in example.get("relevant_sources", [])
            if str(source) in section_catalog
        }
        for section_id in example.get("relevant_sections", []):
            section_id = str(section_id)
            if not any(section_id in section_catalog[source] for source in relevant_sources):
                errors.append(
                    f"{label}: relevant section {section_id!r} not found in its relevant sources"
                )
        gold_evidence = example.get("gold_evidence", [])
        if not isinstance(gold_evidence, list):
            errors.append(f"{label}: gold_evidence must be a list")
            continue
        for evidence in gold_evidence:
            if not isinstance(evidence, dict):
                errors.append(f"{label}: gold_evidence entries must be objects")
                continue
            source = str(evidence.get("source", ""))
            section = str(evidence.get("section", ""))
            if source not in relevant_sources:
                errors.append(
                    f"{label}: gold evidence source {source!r} is not a relevant source"
                )
            elif section not in section_catalog[source]:
                errors.append(
                    f"{label}: gold section {source}#{section} does not exist"
                )
    return errors


if __name__ == "__main__":
    main()
