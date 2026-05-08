from __future__ import annotations

from .schema import Chunk, Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        clean_text = normalize_text(doc.text)
        step = chunk_size - chunk_overlap
        start = 0
        local_index = 0
        while start < len(clean_text):
            end = min(start + chunk_size, len(clean_text))
            text = clean_text[start:end].strip()
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        source=doc.source,
                        chunk_id=f"{doc.source}::chunk-{local_index}",
                        start_char=start,
                        end_char=end,
                    )
                )
                local_index += 1
            if end == len(clean_text):
                break
            start += step
    return chunks


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    compact_lines = [line for line in lines if line]
    return "\n".join(compact_lines)

