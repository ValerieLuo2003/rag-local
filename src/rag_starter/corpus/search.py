from __future__ import annotations

import argparse
import logging
import re

from ..loaders import load_document_file, load_documents, load_manifest


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+#-]*")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search structured corpus sections while authoring gold questions."
    )
    parser.add_argument("query", nargs="+", help="One or more quoted search queries.")
    parser.add_argument("--manifest", default="data/corpus_manifest.jsonl")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context", type=int, default=220)
    args = parser.parse_args()

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    source_filter = set(args.source)
    if source_filter:
        entries = [
            entry
            for entry in load_manifest(args.manifest)
            if entry["doc_id"] in source_filter
        ]
        missing_sources = source_filter - {entry["doc_id"] for entry in entries}
        if missing_sources:
            raise ValueError(f"Unknown source ids: {', '.join(sorted(missing_sources))}")
        documents = [
            load_document_file(entry["local_path"], metadata=entry)
            for entry in entries
        ]
    else:
        documents = load_documents(args.manifest)
    for query in args.query:
        terms = [term.lower() for term in TOKEN_PATTERN.findall(query)]
        if not terms:
            raise ValueError("Each query must contain at least one alphanumeric term.")
        print(f"\n## {query}")
        for result in search_sections(documents, terms, context=args.context)[: args.top_k]:
            coverage, hits, doc_id, section_id, title, page_start, page_end, snippet = result
            page = ""
            if page_start is not None:
                page = (
                    f" pages={page_start}"
                    if page_start == page_end
                    else f" pages={page_start}-{page_end}"
                )
            print(
                f"{doc_id} section={section_id!r} title={title!r}{page} "
                f"coverage={coverage}/{len(terms)} hits={hits}\n  {snippet}"
            )


def search_sections(documents, terms: list[str], *, context: int = 220) -> list[tuple]:
    candidates = []
    for document in documents:
        for section in document.sections:
            normalized = section.text.lower()
            hits = sum(normalized.count(term) for term in terms)
            coverage = sum(term in normalized for term in terms)
            if not hits:
                continue
            first_positions = [normalized.find(term) for term in terms if term in normalized]
            first = min(first_positions) if first_positions else 0
            start = max(0, first - context // 2)
            end = min(len(section.text), start + context)
            snippet = " ".join(section.text[start:end].split())
            candidates.append(
                (
                    coverage,
                    hits,
                    document.doc_id,
                    section.section_id,
                    section.title,
                    section.page_start,
                    section.page_end,
                    snippet,
                )
            )
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return candidates


if __name__ == "__main__":
    main()
