from __future__ import annotations

from pathlib import Path

from .hf_cache import configure_hf_cache
from .schema import SearchResult


class RerankRetriever:
    def __init__(
        self,
        base_retriever,
        candidate_k: int = 50,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        model_cache_dir: str | Path | None = "model_cache",
        max_length: int = 512,
    ) -> None:
        # 先用base_retriever召回候选 再加载cross-encoder reranker精排
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_cache_dir = configure_hf_cache(model_cache_dir)
        self.max_length = max_length
        self.model = self._load_model()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        candidates = self.base_retriever.search(query, top_k=max(self.candidate_k, top_k))
        if not candidates:
            return []

        # cross-encoder的关键： 把（query，chunk）作为一对输入
        pairs = [(query, result.chunk.text) for result in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        # 最后重新排序
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)[:top_k]

        return [
            SearchResult(chunk=result.chunk, score=float(score), rank=rank)
            for rank, (result, score) in enumerate(ranked, start=1)
        ]

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Rerank retrieval requires sentence-transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        cache_folder = str(self.model_cache_dir) if self.model_cache_dir else None
        return CrossEncoder(
            self.model_name,
            cache_folder=cache_folder,
            max_length=self.max_length,
        )

