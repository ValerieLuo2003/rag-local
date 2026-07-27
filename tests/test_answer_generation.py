from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag_starter.answer_generation import (
    AnswerResult,
    extract_citation_indices,
    postprocess_answer,
    response_token_usage,
)


class AnswerGenerationTest(unittest.TestCase):
    def test_citation_parser_uses_explicit_citations_line(self) -> None:
        answer = "FIPS reference [28] appears in the body.\n\nCitations: [1] [3]"
        self.assertEqual(extract_citation_indices(answer), [1, 3])

    def test_postprocessing_preserves_usage_metadata(self) -> None:
        result = AnswerResult(
            answer="Grounded answer. Citations: [1]",
            provider="test",
            model="test-model",
            evidence_count=1,
            latency_ms=12.5,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )

        processed = postprocess_answer(result, evidence_count=1)

        self.assertTrue(processed.citations_valid)
        self.assertEqual(processed.total_tokens, 120)
        self.assertEqual(processed.latency_ms, 12.5)

    def test_chat_and_responses_usage_fields(self) -> None:
        chat_response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        )
        responses_response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16)
        )

        self.assertEqual(response_token_usage(chat_response, chat=True), (10, 4, 14))
        self.assertEqual(response_token_usage(responses_response, chat=False), (11, 5, 16))


if __name__ == "__main__":
    unittest.main()
