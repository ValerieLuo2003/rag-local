from __future__ import annotations

import math
import re
from collections import Counter

from .schema import Chunk, SearchResult


TOKEN_PATTERN = re.compile(r"[\w]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = self._build_idf()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = tokenize(query)
        scored = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)
            if score > 0:
                scored.append((score, chunk))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self.doc_tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))

        idf: dict[str, float] = {}
        for term, freq in document_frequency.items():
            idf[term] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        score = 0.0
        frequencies = self.doc_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]
        for term in query_terms:
            if term not in frequencies:
                continue
            term_frequency = frequencies[term]
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
            )
            score += self.idf.get(term, 0.0) * numerator / denominator
        return score

