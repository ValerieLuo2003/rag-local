from __future__ import annotations

import math
import unittest

from rag_starter.retrieval_metrics import compute_query_metrics, evaluate_retrieval_records


class RetrievalMetricsTest(unittest.TestCase):
    def test_query_metrics_are_source_level_and_ignore_duplicate_chunks(self) -> None:
        metrics = compute_query_metrics(
            retrieved_sources=["noise.md", "a.md", "a.md"],
            relevant_sources=["a.md", "b.md"],
            top_k=3,
        )

        self.assertEqual(metrics.hit_at_k, 1.0)
        self.assertEqual(metrics.recall_at_k, 0.5)
        self.assertEqual(metrics.mrr_at_k, 0.5)
        expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics.ndcg_at_k, expected_ndcg)

    def test_aggregate_excludes_unjudged_queries_and_groups(self) -> None:
        records = [
            {
                "type": "definition",
                "retrieved_sources": ["a.md"],
                "relevant_sources": ["a.md"],
            },
            {
                "type": "comparison",
                "retrieved_sources": ["x.md"],
                "relevant_sources": ["b.md", "c.md"],
            },
            {
                "type": "unanswerable",
                "retrieved_sources": [],
                "relevant_sources": [],
            },
        ]

        report = evaluate_retrieval_records(records, top_k=2, group_by=["type"])

        self.assertEqual(report["overall"]["queries"], 3)
        self.assertEqual(report["overall"]["judged_queries"], 2)
        self.assertEqual(report["overall"]["unjudged_queries"], 1)
        self.assertEqual(report["overall"]["hit_at_k"], 0.5)
        self.assertEqual(report["groups"]["type"]["definition"]["hit_at_k"], 1.0)
        self.assertIsNone(report["groups"]["type"]["unanswerable"]["recall_at_k"])

    def test_invalid_top_k_and_empty_qrels_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            compute_query_metrics(["a.md"], ["a.md"], top_k=0)
        with self.assertRaises(ValueError):
            compute_query_metrics(["a.md"], [], top_k=1)


if __name__ == "__main__":
    unittest.main()
