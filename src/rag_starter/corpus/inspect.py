from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from ..chunking import split_documents
from ..experiment_tracking import write_json
from ..loaders import load_document_file, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect structured corpus extraction quality.")
    parser.add_argument("--manifest", default="data/corpus_manifest.jsonl")
    parser.add_argument("--only-quality-review", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--output-json", default="outputs/corpus_quality.json")
    parser.add_argument("--output-md", default="outputs/corpus_quality.md")
    args = parser.parse_args()

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    entries = load_manifest(args.manifest)
    if args.only_quality_review:
        entries = [entry for entry in entries if entry.get("quality_review") is True]
    if args.limit > 0:
        entries = entries[: args.limit]

    reports = [
        inspect_entry(
            entry,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        for entry in entries
    ]
    summary = {
        "documents": len(reports),
        "status_counts": dict(sorted(Counter(report["status"] for report in reports).items())),
        "source_counts": dict(sorted(Counter(report["source_type"] for report in reports).items())),
        "total_characters": sum(report["characters"] for report in reports),
        "total_sections": sum(report["sections"] for report in reports),
        "total_chunks": sum(report["chunks"] for report in reports),
    }
    payload = {
        "schema_version": 1,
        "manifest": args.manifest,
        "settings": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "only_quality_review": args.only_quality_review,
        },
        "summary": summary,
        "documents": reports,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)

    print(f"documents={summary['documents']}")
    print(f"status_counts={summary['status_counts']}")
    print(f"characters={summary['total_characters']}")
    print(f"sections={summary['total_sections']}")
    print(f"chunks={summary['total_chunks']}")
    print(f"output_json={args.output_json}")
    print(f"output_md={args.output_md}")


def inspect_entry(
    entry: dict,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
) -> dict:
    path = Path(entry["local_path"])
    document = load_document_file(path, metadata=entry)
    chunks = split_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    generic_sections = sum(
        section.section_id == "document"
        or section.section_id == "front-matter"
        or section.section_id.startswith("page-")
        for section in document.sections
    )
    short_sections = sum(len(section.text) < 100 for section in document.sections)
    replacement_characters = document.text.count("\ufffd")
    control_characters = sum(
        ord(character) < 32 and character not in "\n\t\r"
        for character in document.text
    )

    pdf_metrics = inspect_pdf(path) if path.suffix.lower() == ".pdf" else {}
    issues = quality_issues(
        characters=len(document.text),
        sections=len(document.sections),
        chunks=len(chunks),
        generic_section_ratio=generic_sections / len(document.sections) if document.sections else 1.0,
        short_section_ratio=short_sections / len(document.sections) if document.sections else 1.0,
        replacement_characters=replacement_characters,
        control_characters=control_characters,
        pdf_metrics=pdf_metrics,
    )
    status = "pass"
    if any(issue["severity"] == "error" for issue in issues):
        status = "fail"
    elif issues:
        status = "review"

    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "source_type": document.source_type,
        "file": str(path),
        "sha256": document.sha256,
        "characters": len(document.text),
        "sections": len(document.sections),
        "chunks": len(chunks),
        "generic_section_ratio": generic_sections / len(document.sections) if document.sections else 1.0,
        "short_section_ratio": short_sections / len(document.sections) if document.sections else 1.0,
        "replacement_characters": replacement_characters,
        "control_characters": control_characters,
        **pdf_metrics,
        "sample_sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "characters": len(section.text),
            }
            for section in document.sections[:5]
        ],
        "status": status,
        "issues": issues,
    }


def inspect_pdf(path: Path) -> dict:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "PDF quality inspection requires pdfplumber. "
            "Install project dependencies before running this command."
        ) from exc

    page_count = 0
    empty_pages = 0
    total_glyphs = 0
    rotated_glyphs = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            if len(text.strip()) < 40:
                empty_pages += 1
            for character in page.chars:
                total_glyphs += 1
                if character.get("upright") is False:
                    rotated_glyphs += 1
    return {
        "pages": page_count,
        "empty_pages": empty_pages,
        "characters_per_page": None,
        "rotated_glyph_ratio": rotated_glyphs / total_glyphs if total_glyphs else 0.0,
    }


def quality_issues(
    *,
    characters: int,
    sections: int,
    chunks: int,
    generic_section_ratio: float,
    short_section_ratio: float,
    replacement_characters: int,
    control_characters: int,
    pdf_metrics: dict,
) -> list[dict]:
    issues = []

    def add(code: str, severity: str, message: str) -> None:
        issues.append({"code": code, "severity": severity, "message": message})

    if characters < 1000:
        add("too_little_text", "error", f"Only {characters} extracted characters")
    if sections < 2:
        add("too_few_sections", "error", f"Only {sections} detected section(s)")
    if chunks < 2:
        add("too_few_chunks", "error", f"Only {chunks} chunk(s)")
    if generic_section_ratio > 0.5:
        add(
            "weak_section_detection",
            "warning",
            f"{generic_section_ratio:.1%} of sections use generic page/document labels",
        )
    if short_section_ratio > 0.6:
        add(
            "many_short_sections",
            "warning",
            f"{short_section_ratio:.1%} of sections contain fewer than 100 characters",
        )
    if replacement_characters:
        add(
            "replacement_characters",
            "warning",
            f"Found {replacement_characters} Unicode replacement characters",
        )
    if control_characters:
        add(
            "control_characters",
            "warning",
            f"Found {control_characters} unexpected control characters",
        )
    if pdf_metrics:
        pages = int(pdf_metrics["pages"])
        empty_pages = int(pdf_metrics["empty_pages"])
        rotated_ratio = float(pdf_metrics["rotated_glyph_ratio"])
        if pages and empty_pages / pages > 0.1:
            add(
                "empty_pdf_pages",
                "warning",
                f"{empty_pages}/{pages} pages have fewer than 40 extracted characters",
            )
        if rotated_ratio > 0.01:
            add(
                "rotated_text",
                "warning",
                f"{rotated_ratio:.1%} of PDF glyphs are rotated; tables may be incomplete",
            )
    return issues


def write_markdown(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Corpus Extraction Quality Report",
        "",
        f"- Documents: {summary['documents']}",
        f"- Sources: {summary['source_counts']}",
        f"- Status: {summary['status_counts']}",
        f"- Characters: {summary['total_characters']}",
        f"- Sections: {summary['total_sections']}",
        f"- Chunks: {summary['total_chunks']}",
        "",
        "| Document | Source | Chars | Sections | Chunks | Pages | Rotated glyphs | Status | Issues |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for report in payload["documents"]:
        issues = "; ".join(issue["code"] for issue in report["issues"]) or "-"
        pages = report.get("pages", "-")
        rotated = report.get("rotated_glyph_ratio")
        rotated_text = f"{rotated:.2%}" if isinstance(rotated, float) else "-"
        lines.append(
            f"| {report['doc_id']} | {report['source_type']} | {report['characters']} | "
            f"{report['sections']} | {report['chunks']} | {pages} | {rotated_text} | "
            f"{report['status']} | {issues} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pass`: no automatic extraction warning.",
            "- `review`: usable text, but a human should inspect the flagged layout/section issue.",
            "- `fail`: insufficient text, sections, or chunks for retrieval evaluation.",
            "- Rotated glyph warnings usually indicate tables or side labels; do not use those chunks as gold evidence without visual review.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
