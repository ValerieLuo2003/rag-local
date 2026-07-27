from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from rag_starter.eval_retrieval import load_eval_set
from rag_starter.validate_eval_set import validate_examples


class ValidateEvalSetTest(unittest.TestCase):
    def test_valid_domain_examples_pass(self) -> None:
        examples = [
            {
                "query_id": "hash-001",
                "question": "What is collision resistance?",
                "query_type": "definition",
                "answerable": True,
                "reference_answer": "A normalized reference answer.",
                "relevant_sources": ["standard.pdf"],
            },
            {
                "query_id": "hash-002",
                "question": "A deliberately unanswerable question.",
                "query_type": "unanswerable",
                "answerable": False,
                "reference_answer": "Insufficient evidence.",
                "relevant_sources": [],
            },
        ]

        errors = validate_examples(
            examples,
            min_examples=2,
            require_reference_answer=True,
        )

        self.assertEqual(errors, [])

    def test_missing_labels_are_reported(self) -> None:
        errors = validate_examples(
            [{"question": "Incomplete", "relevant_sources": []}],
            min_examples=1,
        )

        self.assertTrue(any("query_id" in error for error in errors))
        self.assertTrue(any("query_type" in error for error in errors))
        self.assertTrue(any("answerable" in error for error in errors))

    def test_verified_labels_are_required_when_requested(self) -> None:
        errors = validate_examples(
            [
                {
                    "query_id": "q1",
                    "question": "Question",
                    "query_type": "definition",
                    "answerable": True,
                    "reference_answer": "Answer",
                    "relevant_sources": ["source"],
                }
            ],
            min_examples=1,
            require_verified=True,
        )

        self.assertTrue(any("review_status" in error for error in errors))
        self.assertTrue(any("reviewer" in error for error in errors))

    def test_eval_manifest_combines_jsonl_parts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jsonl").write_text(
                json.dumps({"query_id": "a", "question": "A"}) + "\n",
                encoding="utf-8",
            )
            (root / "b.jsonl").write_text(
                json.dumps({"query_id": "b", "question": "B"}) + "\n",
                encoding="utf-8",
            )
            (root / "eval.json").write_text(
                json.dumps(
                    {
                        "parts": ["a.jsonl", "b.jsonl"],
                        "expected_examples": 2,
                    }
                ),
                encoding="utf-8",
            )

            examples = load_eval_set(root / "eval.json")

        self.assertEqual([example["query_id"] for example in examples], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
