from __future__ import annotations

import unittest

from rag_starter.corpus.inspect import quality_issues


class CorpusInspectTest(unittest.TestCase):
    def test_clean_document_has_no_issues(self) -> None:
        issues = quality_issues(
            characters=5000,
            sections=10,
            chunks=12,
            generic_section_ratio=0.1,
            short_section_ratio=0.2,
            replacement_characters=0,
            control_characters=0,
            pdf_metrics={"pages": 10, "empty_pages": 0, "rotated_glyph_ratio": 0.0},
        )
        self.assertEqual(issues, [])

    def test_rotated_text_and_insufficient_content_are_flagged(self) -> None:
        issues = quality_issues(
            characters=500,
            sections=1,
            chunks=1,
            generic_section_ratio=1.0,
            short_section_ratio=1.0,
            replacement_characters=2,
            control_characters=0,
            pdf_metrics={"pages": 10, "empty_pages": 2, "rotated_glyph_ratio": 0.05},
        )
        codes = {issue["code"] for issue in issues}
        self.assertIn("too_little_text", codes)
        self.assertIn("rotated_text", codes)
        self.assertIn("empty_pdf_pages", codes)


if __name__ == "__main__":
    unittest.main()
