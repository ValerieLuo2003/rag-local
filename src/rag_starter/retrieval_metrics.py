from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class QueryRetrievalMetrics:
    """Binary, source-level retrieval metrics for one query."""

    hit_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


def compute_query_metrics(
    retrieved_sources: Sequence[str],
    relevant_sources: Iterable[str],
    top_k: int,
) -> QueryRetrievalMetrics:
    """Compute source-level metrics.

    A source contributes relevance gain only on its first occurrence. This prevents
    several chunks from the same document from inflating Recall or nDCG.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    relevant = {str(source) for source in relevant_sources if str(source)}
    if not relevant:
        raise ValueError("relevant_sources must contain at least one source")

    first_relevant_rank: int | None = None
    retrieved_relevant: set[str] = set()
    dcg = 0.0

    for rank, source in enumerate(retrieved_sources[:top_k], start=1):
        if source not in relevant or source in retrieved_relevant:
            continue
        retrieved_relevant.add(source)
        if first_relevant_rank is None:
            first_relevant_rank = rank
        dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), top_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return QueryRetrievalMetrics(
        hit_at_k=float(bool(retrieved_relevant)),
        recall_at_k=len(retrieved_relevant) / len(relevant),
        mrr_at_k=1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0,
        ndcg_at_k=dcg / idcg if idcg else 0.0,
    )


def evaluate_retrieval_records(
    records: Sequence[Mapping[str, object]],
    top_k: int,
    group_by: Sequence[str] = (),
) -> dict:
    """Aggregate retrieval records overall and by query metadata fields.

    Each record must contain ``retrieved_sources`` and may contain
    ``relevant_sources``. Records without relevance judgments are counted but
    excluded from relevance-metric denominators.
    """

    overall = _aggregate(records, top_k)
    grouped: dict[str, dict[str, dict]] = {}

    for field in group_by:
        field = field.strip()
        if not field:
            continue
        buckets: dict[str, list[Mapping[str, object]]] = {}
        for record in records:
            value = record.get(field, "unknown")
            label = str(value) if value not in (None, "") else "unknown"
            buckets.setdefault(label, []).append(record)
        grouped[field] = {
            label: _aggregate(bucket_records, top_k)
            for label, bucket_records in sorted(buckets.items())
        }

    return {
        "top_k": top_k,
        "overall": overall,
        "groups": grouped,
    }


def _aggregate(records: Sequence[Mapping[str, object]], top_k: int) -> dict:
    judged_metrics: list[QueryRetrievalMetrics] = []
    for record in records:
        relevant_sources = _as_string_list(record.get("relevant_sources"))
        if not relevant_sources:
            continue
        retrieved_sources = _as_string_list(record.get("retrieved_sources"))
        judged_metrics.append(compute_query_metrics(retrieved_sources, relevant_sources, top_k))

    metric_names = tuple(QueryRetrievalMetrics.__dataclass_fields__)
    result = {
        "queries": len(records),
        "judged_queries": len(judged_metrics),
        "unjudged_queries": len(records) - len(judged_metrics),
    }
    for metric_name in metric_names:
        values = [getattr(item, metric_name) for item in judged_metrics]
        result[metric_name] = mean(values) if values else None
    return result


def query_metrics_as_dict(
    retrieved_sources: Sequence[str],
    relevant_sources: Iterable[str],
    top_k: int,
) -> dict[str, float]:
    return asdict(compute_query_metrics(retrieved_sources, relevant_sources, top_k))


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None]
    raise TypeError(f"Expected a string or iterable of strings, got {type(value).__name__}")
