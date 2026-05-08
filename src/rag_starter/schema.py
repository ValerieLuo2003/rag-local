from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_id: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

