from __future__ import annotations

import unittest

from rag_starter.generation_metrics import aggregate_generation_records


class GenerationMetricsTest(unittest.TestCase):
    def test_refusal_metrics_and_human_scores(self) -> None:
        records = [
            {
                "query_type": "answerable",
                "answerable": True,
                "refused": False,
                "citations_valid": True,
                "citation_source_precision": 1.0,
                "evaluation": {
                    "answer_correctness": 0.8,
                    "citation_correctness": 1.0,
                    "citation_completeness": 0.5,
                    "faithfulness": 1.0,
                },
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "estimated_cost_usd": 0.01,
                "retrieval_latency_ms": 10,
                "generation_latency_ms": 50,
                "total_latency_ms": 60,
            },
            {
                "query_type": "unanswerable",
                "answerable": False,
                "refused": True,
                "citations_valid": True,
                "retrieval_latency_ms": 20,
                "generation_latency_ms": 0,
                "total_latency_ms": 20,
            },
            {
                "query_type": "unanswerable",
                "answerable": False,
                "refused": False,
                "citations_valid": False,
            },
        ]

        overall = aggregate_generation_records(records)["overall"]

        self.assertEqual(overall["citation_format_valid_rate"], 0.5)
        self.assertEqual(overall["refusal"]["precision"], 1.0)
        self.assertEqual(overall["refusal"]["recall"], 0.5)
        self.assertAlmostEqual(overall["refusal"]["f1"], 2 / 3)
        self.assertEqual(
            overall["human_or_judge_scores"]["answer_correctness"]["mean"],
            0.8,
        )
        self.assertEqual(overall["tokens"]["total"], 120)
        self.assertEqual(overall["cost_usd"]["total"], 0.01)

    def test_dry_run_is_not_scored_as_an_answer(self) -> None:
        records = [
            {
                "query_type": "test",
                "answerable": True,
                "refused": False,
                "citations_valid": False,
                "dry_run": True,
            }
        ]

        overall = aggregate_generation_records(records)["overall"]

        self.assertEqual(overall["evaluation_eligible_queries"], 0)
        self.assertIsNone(overall["citation_format_valid_rate"])
        self.assertEqual(overall["refusal"]["labeled_queries"], 0)

    def test_human_scores_must_be_normalized(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_generation_records(
                [{"evaluation": {"faithfulness": 2.0}, "refused": False}]
            )


if __name__ == "__main__":
    unittest.main()
