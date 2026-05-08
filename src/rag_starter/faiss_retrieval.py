from __future__ import annotations

from pathlib import Path

from .embedding_retrieval import EmbeddingRetriever
from .schema import Chunk, SearchResult


class FaissRetriever(EmbeddingRetriever):
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_path: str | Path | None = None,
        model_cache_dir: str | Path | None = "model_cache",
    ) -> None:
        super().__init__(
            chunks=chunks,
            model_name=model_name,
            batch_size=batch_size,
            cache_path=cache_path,
            model_cache_dir=model_cache_dir,
        )
        self.index = self._build_index()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            SearchResult(
                chunk=self.chunks[int(index)],
                score=float(score),
                rank=rank,
            )
            for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1)
            if int(index) >= 0
        ]

    def _build_index(self):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS retrieval requires faiss-cpu. Install with: pip install faiss-cpu") from exc

        embeddings = self.embeddings.astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

