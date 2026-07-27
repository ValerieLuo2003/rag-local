from __future__ import annotations

from statistics import mean
from typing import Mapping, Sequence

from .experiment_tracking import latency_summary


HUMAN_SCORE_FIELDS = (
    "answer_correctness",
    "citation_correctness",
    "citation_completeness",
    "faithfulness",
)


def aggregate_generation_records(
    records: Sequence[Mapping[str, object]],
    group_by: str | None = "query_type",
) -> dict:
    result = {"overall": _aggregate(records)}
    if group_by:
        buckets: dict[str, list[Mapping[str, object]]] = {}
        for record in records:
            value = record.get(group_by, "unknown")
            label = str(value) if value not in (None, "") else "unknown"
            buckets.setdefault(label, []).append(record)
        result["group_by"] = group_by
        result["groups"] = {
            label: _aggregate(bucket_records)
            for label, bucket_records in sorted(buckets.items())
        }
    else:
        result["group_by"] = None
        result["groups"] = {}
    return result


def _aggregate(records: Sequence[Mapping[str, object]]) -> dict:
    eligible = [record for record in records if not bool(record.get("dry_run"))]
    answered = [record for record in eligible if not bool(record.get("refused"))]
    citation_format_values = [
        float(bool(record["citations_valid"]))
        for record in answered
        if record.get("citations_valid") is not None
    ]
    citation_source_precision = _numeric_values(eligible, "citation_source_precision")
    citation_source_recall = _numeric_values(eligible, "citation_source_recall")

    result = {
        "queries": len(records),
        "evaluation_eligible_queries": len(eligible),
        "answered_queries": len(answered),
        "refused_queries": sum(bool(record.get("refused")) for record in eligible),
        "citation_format_valid_rate": _mean_or_none(citation_format_values),
        "citation_format_labeled": len(citation_format_values),
        "citation_source_precision_proxy": _mean_or_none(citation_source_precision),
        "citation_source_precision_labeled": len(citation_source_precision),
        "citation_source_recall_proxy": _mean_or_none(citation_source_recall),
        "citation_source_recall_labeled": len(citation_source_recall),
        "refusal": _refusal_metrics(eligible),
        "human_or_judge_scores": _human_score_summary(eligible),
        "latency_ms": {
            field.removesuffix("_ms"): latency_summary(_numeric_values(eligible, field))
            for field in ("retrieval_latency_ms", "generation_latency_ms", "total_latency_ms")
        },
        "tokens": _token_summary(eligible),
        "cost_usd": _cost_summary(eligible),
    }
    return result


def _refusal_metrics(records: Sequence[Mapping[str, object]]) -> dict:
    labeled = [record for record in records if isinstance(record.get("answerable"), bool)]
    true_positive = false_positive = false_negative = true_negative = 0
    for record in labeled:
        expected_refusal = not bool(record["answerable"])
        predicted_refusal = bool(record.get("refused"))
        if predicted_refusal and expected_refusal:
            true_positive += 1
        elif predicted_refusal and not expected_refusal:
            false_positive += 1
        elif not predicted_refusal and expected_refusal:
            false_negative += 1
        else:
            true_negative += 1

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "labeled_queries": len(labeled),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _human_score_summary(records: Sequence[Mapping[str, object]]) -> dict:
    summary = {}
    for field in HUMAN_SCORE_FIELDS:
        values = []
        for record in records:
            evaluation = record.get("evaluation")
            if not isinstance(evaluation, Mapping):
                continue
            value = evaluation.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not 0.0 <= numeric <= 1.0:
                    raise ValueError(f"evaluation.{field} must be in [0, 1], got {numeric}")
                values.append(numeric)
        summary[field] = {
            "mean": _mean_or_none(values),
            "labeled_queries": len(values),
        }
    return summary


def _token_summary(records: Sequence[Mapping[str, object]]) -> dict:
    input_tokens = _numeric_values(records, "input_tokens")
    output_tokens = _numeric_values(records, "output_tokens")
    total_tokens = _numeric_values(records, "total_tokens")
    estimated_prompt_tokens = _numeric_values(records, "estimated_prompt_tokens")
    return {
        "actual_usage_queries": len(total_tokens),
        "input_total": int(sum(input_tokens)),
        "input_mean": _mean_or_none(input_tokens),
        "output_total": int(sum(output_tokens)),
        "output_mean": _mean_or_none(output_tokens),
        "total": int(sum(total_tokens)),
        "mean": _mean_or_none(total_tokens),
        "estimated_prompt_mean": _mean_or_none(estimated_prompt_tokens),
    }


def _cost_summary(records: Sequence[Mapping[str, object]]) -> dict:
    values = _numeric_values(records, "estimated_cost_usd")
    return {
        "priced_queries": len(values),
        "total": sum(values) if values else None,
        "mean": _mean_or_none(values),
    }


def _numeric_values(records: Sequence[Mapping[str, object]], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None
