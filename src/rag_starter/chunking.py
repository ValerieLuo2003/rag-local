from __future__ import annotations

from .schema import Chunk, Document, DocumentSection


def split_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    validate_chunking(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for document in documents:
        sections = document.sections or (
            DocumentSection(
                section_id="document",
                title=document.title or "Document",
                text=document.text,
                start_char=0,
                end_char=len(document.text),
            ),
        )
        local_index = 0
        for section in sections:
            for local_start, local_end, text in split_section_text(
                section.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                chunks.append(
                    Chunk(
                        text=text,
                        source=document.source,
                        chunk_id=(
                            f"{document.doc_id or document.source}::"
                            f"{section.section_id}::chunk-{local_index}"
                        ),
                        start_char=section.start_char + local_start,
                        end_char=section.start_char + local_end,
                        doc_id=document.doc_id or document.source,
                        title=document.title,
                        source_type=document.source_type,
                        version=document.version,
                        status=document.status,
                        canonical_url=document.canonical_url,
                        supersedes=document.supersedes,
                        superseded_by=document.superseded_by,
                        topics=document.topics,
                        section_id=section.section_id,
                        section_title=section.title,
                        page_start=section.page_start,
                        page_end=section.page_end,
                    )
                )
                local_index += 1
    return chunks


def split_section_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int, str]]:
    validate_chunking(chunk_size, chunk_overlap)
    clean_text = normalize_text(text)
    if not clean_text:
        return []

    chunks = []
    start = 0
    while start < len(clean_text):
        proposed_end = min(start + chunk_size, len(clean_text))
        end = choose_boundary(clean_text, start, proposed_end, chunk_size)
        chunk_text = clean_text[start:end].strip()
        if chunk_text:
            leading = len(clean_text[start:end]) - len(clean_text[start:end].lstrip())
            trailing = len(clean_text[start:end]) - len(clean_text[start:end].rstrip())
            chunks.append((start + leading, end - trailing, chunk_text))
        if end >= len(clean_text):
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start
    return chunks


def choose_boundary(text: str, start: int, proposed_end: int, chunk_size: int) -> int:
    if proposed_end >= len(text):
        return len(text)
    minimum = start + max(int(chunk_size * 0.65), 1)
    for separator in ("\n", ". ", "; ", ", ", " "):
        boundary = text.rfind(separator, minimum, proposed_end)
        if boundary >= minimum:
            return boundary + len(separator)
    return proposed_end


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    compact_lines = [line for line in lines if line]
    return "\n".join(compact_lines)
