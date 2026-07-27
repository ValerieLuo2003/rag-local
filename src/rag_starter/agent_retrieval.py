from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .schema import Chunk, SearchResult


CURRENT_VERSION_CUES = (
    "当前",
    "现行",
    "最新",
    "current",
    "latest",
    "supersed",
    "废止",
    "取代",
)
COMPARISON_CUES = (
    "对比",
    "比较",
    "相比",
    "区别",
    "分别",
    "结合",
    "对照",
    "跨文档",
    "共同",
    "versus",
    " vs ",
    "compare",
)
CLAUSE_SPLIT = re.compile(r"[；;?？]|\bversus\b|\bvs\.?\b", flags=re.IGNORECASE)
NORMALIZE_ID = re.compile(r"[^a-z0-9]+")
NIST_REVISION_SUFFIX = re.compile(r"(?:r\d+|upd\d+|pt\d+).*$", flags=re.IGNORECASE)
TOPIC_CUES = {
    "aes": ("aes",),
    "chacha20": ("chacha20",),
    "cose": ("cose",),
    "openpgp": ("openpgp",),
    "argon2": ("argon2",),
    "hash-to-curve": ("hash-to-curve", "hash to curve", "清除余因子"),
    "security-strength": ("security strength", "安全强度"),
    "hkdf": ("hkdf",),
    "ml-kem": ("ml-kem", "ml kem"),
    "ml-dsa": ("ml-dsa", "ml dsa"),
    "slh-dsa": ("slh-dsa", "slh dsa"),
}


@dataclass(frozen=True)
class QueryPlan:
    query: str
    subqueries: tuple[str, ...]
    explicit_sources: tuple[str, ...]
    routed_sources: tuple[str, ...]
    comparison: bool
    current_version: bool

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "subqueries": list(self.subqueries),
            "explicit_sources": list(self.explicit_sources),
            "routed_sources": list(self.routed_sources),
            "comparison": self.comparison,
            "current_version": self.current_version,
        }


class AgenticRetriever:
    """Rule-based retrieval agent for auditable domain routing.

    The agent deliberately does not call an LLM. Its plan is deterministic,
    serializable, and suitable for ablation against the base retriever.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        base_retriever,
        *,
        candidate_k: int = 40,
        rrf_k: int = 60,
        max_per_source: int = 2,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if max_per_source <= 0:
            raise ValueError("max_per_source must be positive")
        self.chunks = chunks
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.max_per_source = max_per_source
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.source_chunks: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            self.source_chunks[chunk.doc_id or chunk.source].append(chunk)
        self.source_aliases = build_source_aliases(chunks)
        self.topic_sources = build_topic_sources(chunks)
        self.last_plan: dict = {}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        plan = self.plan(query)
        self.last_plan = plan.as_dict()
        fused_scores: dict[str, float] = defaultdict(float)

        for subquery in plan.subqueries:
            results = self.base_retriever.search(
                subquery,
                top_k=max(self.candidate_k, top_k),
            )
            for result in results:
                fused_scores[result.chunk.chunk_id] += 1.0 / (self.rrf_k + result.rank)

        for chunk_id in list(fused_scores):
            chunk = self.chunk_by_id[chunk_id]
            source = chunk.doc_id or chunk.source
            if source in plan.explicit_sources:
                fused_scores[chunk_id] += 0.04
            elif source in plan.routed_sources:
                fused_scores[chunk_id] += 0.025
            if plan.current_version:
                if chunk.superseded_by or chunk.status.lower() == "obsolete":
                    fused_scores[chunk_id] *= 0.35
                elif chunk.supersedes:
                    fused_scores[chunk_id] += 0.01

        ranked_chunks = sorted(
            fused_scores,
            key=lambda chunk_id: (
                fused_scores[chunk_id],
                self.chunk_by_id[chunk_id].doc_id,
                chunk_id,
            ),
            reverse=True,
        )
        selected = self._select(
            ranked_chunks,
            top_k=top_k,
            plan=plan,
        )
        return [
            SearchResult(
                chunk=self.chunk_by_id[chunk_id],
                score=fused_scores[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(selected, start=1)
        ]

    def plan(self, query: str) -> QueryPlan:
        normalized_query = normalize_identifier(query)
        explicit_sources = tuple(
            source
            for source, aliases in self.source_aliases.items()
            if any(alias and alias in normalized_query for alias in aliases)
        )
        lowered = query.lower()
        matched_topics = {
            topic
            for topic, cues in TOPIC_CUES.items()
            if any(cue in lowered for cue in cues)
        }
        routed_sources = tuple(
            source
            for source in self.source_chunks
            if any(
                source in self.topic_sources.get(topic, ())
                for topic in matched_topics
            )
            and source not in explicit_sources
        )
        comparison = (
            len(explicit_sources) >= 2
            or any(cue in lowered for cue in COMPARISON_CUES)
        )
        current_version = any(cue in lowered for cue in CURRENT_VERSION_CUES)

        subqueries = [query.strip()]
        for clause in CLAUSE_SPLIT.split(query):
            clause = clause.strip(" ，,。")
            if len(clause) >= 8 and clause not in subqueries:
                subqueries.append(clause)
        for source in (*explicit_sources, *routed_sources):
            exemplar = self.source_chunks[source][0]
            source_query = f"{query} {source} {exemplar.title}".strip()
            if source_query not in subqueries:
                subqueries.append(source_query)
        if current_version:
            for source, chunks in self.source_chunks.items():
                exemplar = chunks[0]
                if source in explicit_sources and exemplar.superseded_by:
                    successor = exemplar.superseded_by
                    if successor in self.source_chunks:
                        successor_title = self.source_chunks[successor][0].title
                        subqueries.append(f"{query} {successor} {successor_title}")

        return QueryPlan(
            query=query,
            subqueries=tuple(dict.fromkeys(subqueries)),
            explicit_sources=explicit_sources,
            routed_sources=routed_sources,
            comparison=comparison,
            current_version=current_version,
        )

    def _select(
        self,
        ranked_chunks: list[str],
        *,
        top_k: int,
        plan: QueryPlan,
    ) -> list[str]:
        selected: list[str] = []
        source_counts: dict[str, int] = defaultdict(int)

        # Cross-document questions first reserve one evidence slot for every
        # explicitly named document that has a candidate.
        if plan.comparison:
            explicit_source_set = set((*plan.explicit_sources, *plan.routed_sources))
            reserved_sources: set[str] = set()
            for candidate in ranked_chunks:
                source = (
                    self.chunk_by_id[candidate].doc_id
                    or self.chunk_by_id[candidate].source
                )
                if source in explicit_source_set and source not in reserved_sources:
                    selected.append(candidate)
                    source_counts[source] += 1
                    reserved_sources.add(source)
                    if len(selected) >= top_k:
                        return selected
                if len(reserved_sources) >= len(explicit_source_set):
                    break

        cap = self.max_per_source if plan.comparison else top_k
        for chunk_id in ranked_chunks:
            if chunk_id in selected:
                continue
            source = self.chunk_by_id[chunk_id].doc_id or self.chunk_by_id[chunk_id].source
            if source_counts[source] >= cap:
                continue
            selected.append(chunk_id)
            source_counts[source] += 1
            if len(selected) >= top_k:
                break

        # A strict diversity cap should not return fewer than top_k results.
        if len(selected) < top_k:
            for chunk_id in ranked_chunks:
                if chunk_id not in selected:
                    selected.append(chunk_id)
                    if len(selected) >= top_k:
                        break
        return selected


def build_source_aliases(chunks: list[Chunk]) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        source = chunk.doc_id or chunk.source
        normalized_source = normalize_identifier(source)
        aliases[source].add(normalized_source)
        aliases[source].add(normalize_identifier(source.removeprefix("NIST.")))
        if source.startswith("RFC."):
            number = source.split(".", 1)[1]
            aliases[source].add(normalize_identifier(f"RFC {number}"))
        if source.startswith("NIST.FIPS."):
            number = source.removeprefix("NIST.FIPS.")
            aliases[source].add(normalize_identifier(f"FIPS {number}"))
            base_number = NIST_REVISION_SUFFIX.sub("", number).rstrip("-.")
            aliases[source].add(normalize_identifier(f"FIPS {base_number}"))
        if source.startswith("NIST.SP."):
            number = source.removeprefix("NIST.SP.")
            aliases[source].add(normalize_identifier(f"SP {number}"))
            base_number = NIST_REVISION_SUFFIX.sub("", number).rstrip("-.")
            aliases[source].add(normalize_identifier(f"SP {base_number}"))
    return {
        source: tuple(sorted(values, key=len, reverse=True))
        for source, values in aliases.items()
    }


def build_topic_sources(chunks: list[Chunk]) -> dict[str, tuple[str, ...]]:
    topic_sources: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        source = chunk.doc_id or chunk.source
        for topic in chunk.topics:
            topic_sources[topic.lower()].add(source)
    return {
        topic: tuple(sorted(sources))
        for topic, sources in topic_sources.items()
    }


def normalize_identifier(value: str) -> str:
    return NORMALIZE_ID.sub("", value.lower())
