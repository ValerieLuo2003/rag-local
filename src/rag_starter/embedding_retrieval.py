from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .hf_cache import configure_hf_cache
from .schema import Chunk, SearchResult


class EmbeddingRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_path: str | Path | None = None,
        model_cache_dir: str | Path | None = "model_cache",
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.model_cache_dir = configure_hf_cache(model_cache_dir)
        self.model = self._load_model(model_name)
        self.embeddings = self._load_or_encode_embeddings()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import numpy as np

        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[int(index)],
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(top_indices, start=1)
        ]

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding retrieval requires sentence-transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        cache_folder = str(self.model_cache_dir) if self.model_cache_dir else None
        return SentenceTransformer(model_name, cache_folder=cache_folder)

    def _load_or_encode_embeddings(self):
        import numpy as np

        # 先算一个fingerprint 根据chunk内容生成一个哈希 如果文档没变 就直接读取.npz缓存 如果文档变了 就重新编码
        fingerprint = self._fingerprint()
        if self.cache_path and self.cache_path.exists():
            try:
                cached = np.load(self.cache_path, allow_pickle=False)
                metadata = json.loads(str(cached["metadata"]))
                if metadata.get("fingerprint") == fingerprint and metadata.get("model_name") == self.model_name:
                    return cached["embeddings"]
            except (OSError, EOFError, ValueError, KeyError, json.JSONDecodeError):
                pass

        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "fingerprint": fingerprint,
                "model_name": self.model_name,
                "chunk_count": len(self.chunks),
            }
            temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp.npz")
            np.savez_compressed(
                temp_path,
                embeddings=embeddings,
                metadata=json.dumps(metadata, ensure_ascii=False),
            )
            temp_path.replace(self.cache_path)

        return embeddings

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(str(chunk.start_char).encode("utf-8"))
            digest.update(str(chunk.end_char).encode("utf-8"))
            digest.update(chunk.text.encode("utf-8"))
        return digest.hexdigest()
