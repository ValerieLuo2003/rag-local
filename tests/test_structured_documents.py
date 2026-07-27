from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_starter.chunking import split_documents
from rag_starter.loaders import (
    load_document_file,
    match_section_heading,
    read_rfc_xml_sections,
    read_text_sections,
    sections_from_pdf_pages,
)
from rag_starter.schema import Document


class StructuredDocumentsTest(unittest.TestCase):
    def test_markdown_sections_are_preserved_in_chunks(self) -> None:
        sections = read_text_sections(
            "# First Section\nAlpha evidence.\n\n# Second Section\nBeta evidence."
        )
        text = "\n\n".join(section.text for section in sections)
        document = Document(
            text=text,
            source="demo",
            doc_id="demo",
            title="Demo",
            sections=tuple(sections),
        )

        chunks = split_documents([document], chunk_size=50, chunk_overlap=5)

        self.assertEqual({chunk.section_id for chunk in chunks}, {"first-section", "second-section"})
        self.assertFalse(any("Alpha evidence" in chunk.text and "Beta evidence" in chunk.text for chunk in chunks))

    def test_rfc_xml_sections_keep_hierarchy(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<rfc>
  <front><abstract><t>Abstract text.</t></abstract></front>
  <middle>
    <section name="Introduction" anchor="intro">
      <t>Intro text.</t>
      <section name="Child"><t>Child text.</t></section>
    </section>
  </middle>
</rfc>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rfc-test.xml"
            path.write_text(xml, encoding="utf-8")
            sections = read_rfc_xml_sections(path)

        self.assertEqual([section.section_id for section in sections], ["abstract", "intro", "1.1"])
        self.assertEqual(sections[1].text, "Intro text.")
        self.assertEqual(sections[2].level, 2)

    def test_manifest_metadata_flows_to_citation_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standard.md"
            path.write_text("# Scope\nAuthoritative text.", encoding="utf-8")
            document = load_document_file(
                path,
                metadata={
                    "doc_id": "TEST.STANDARD.1",
                    "title": "Test Standard",
                    "source_type": "nist",
                    "version": "2026",
                    "status": "final",
                    "canonical_url": "https://example.test/standard",
                    "download_url": "https://example.test/standard.pdf",
                },
            )

        chunk = split_documents([document], chunk_size=100, chunk_overlap=10)[0]
        self.assertEqual(chunk.doc_id, "TEST.STANDARD.1")
        self.assertEqual(chunk.section_id, "scope")
        self.assertIn("Test Standard (2026)", chunk.citation_label())
        self.assertIn("Section scope", chunk.citation_label())

    def test_pdf_heading_filter_rejects_cover_metadata(self) -> None:
        self.assertIsNone(match_section_heading("100 Bureau Drive (Mail Stop 8930)"))
        self.assertIsNone(
            match_section_heading(
                "3 Explanation: This Standard specifies secure hash algorithms and requirements"
            )
        )
        self.assertIsNone(match_section_heading("Appendix B to facilitate implementations"))
        self.assertIsNone(match_section_heading("3 SP 800-108 Recommendation for Key Derivation"))
        self.assertIsNotNone(match_section_heading("5.1 Encryption Function"))

    def test_rfc_text_requires_dotted_heading_identifier(self) -> None:
        sections = read_text_sections(
            "1. Introduction\nIntro.\n2 Message authentication is useful\n"
            "Body.\n   2. Indented list item\n2. Definition of HMAC\nDefinition.\n"
            "# Code comment, not a heading\nA.1.  ffdhe2048\nParameters.",
            require_dotted_headings=True,
        )

        self.assertEqual([section.section_id for section in sections], ["1", "2", "A.1"])
        self.assertIn("2 Message authentication is useful", sections[0].text)
        self.assertIn("2. Indented list item", sections[0].text)
        self.assertIn("# Code comment", sections[1].text)

    def test_pdf_falls_back_to_page_sections_when_front_matter_dominates(self) -> None:
        sections = sections_from_pdf_pages(
            [
                (1, "Cover"),
                (2, "Contents"),
                (3, "More front matter"),
                (4, "1 Actual Section\nBody"),
                (5, "2 Next Section\nBody"),
            ]
        )

        self.assertEqual([section.section_id for section in sections], [
            "page-1",
            "page-2",
            "page-3",
            "page-4",
            "page-5",
        ])


if __name__ == "__main__":
    unittest.main()
