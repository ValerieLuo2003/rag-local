from __future__ import annotations

from .embedding_retrieval import EmbeddingRetriever
from .retrieval import BM25Retriever
from .schema import Chunk, SearchResult


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_path: str | None = None,
        model_cache_dir: str | None = "model_cache",
        candidate_k: int = 50,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        embedding_weight: float = 1.0,
    ) -> None:
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        # 同时创建两个检索器
        self.bm25 = BM25Retriever(chunks)
        self.embedding = EmbeddingRetriever(
            chunks,
            model_name=model_name,
            batch_size=batch_size,
            cache_path=cache_path,
            model_cache_dir=model_cache_dir,
        )
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        candidate_k = max(self.candidate_k, top_k)
        # 搜索时先分别检索
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        embedding_results = self.embedding.search(query, top_k=candidate_k)

        fused_scores: dict[str, float] = {}
        # 用RRF融合
        # 排名越靠前，加分越多，同时被BM25和embedding找到，加分会叠加
        for result in bm25_results:
            fused_scores[result.chunk.chunk_id] = fused_scores.get(result.chunk.chunk_id, 0.0) + (
                self.bm25_weight / (self.rrf_k + result.rank)
            )
        for result in embedding_results:
            fused_scores[result.chunk.chunk_id] = fused_scores.get(result.chunk.chunk_id, 0.0) + (
                self.embedding_weight / (self.rrf_k + result.rank)
            )

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=self.chunk_by_id[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

