from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_starter.corpus.download import (
    validate_download_url,
    validate_entry_url,
    validate_local_payload,
)


class CorpusDownloadTest(unittest.TestCase):
    def test_source_specific_whitelist(self) -> None:
        validate_download_url(
            "https://nvlpubs.nist.gov/nistpubs/FIPS/example.pdf",
            "nist",
        )
        validate_download_url(
            "https://www.rfc-editor.org/rfc/rfc9180.xml",
            "rfc",
        )
        with self.assertRaises(ValueError):
            validate_download_url("https://example.com/paper.pdf", "nist")
        with self.assertRaises(ValueError):
            validate_download_url("http://www.rfc-editor.org/rfc/rfc9180.xml", "rfc")

    def test_entry_source_and_host_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            validate_entry_url(
                {
                    "source_type": "rfc",
                    "download_url": "https://nvlpubs.nist.gov/example.pdf",
                }
            )

    def test_payload_magic_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "standard.pdf"
            xml = Path(directory) / "rfc.xml"
            text = Path(directory) / "rfc.txt"
            bad = Path(directory) / "error.pdf"
            pdf.write_bytes(b"%PDF-1.7\nexample")
            xml.write_bytes(b'<?xml version="1.0"?><rfc></rfc>')
            text.write_bytes(b"Network Working Group\nRequest for Comments: 9999\n")
            bad.write_bytes(b"<html>error</html>")

            validate_local_payload(pdf, "nist")
            validate_local_payload(xml, "rfc")
            validate_local_payload(text, "rfc")
            with self.assertRaises(ValueError):
                validate_local_payload(bad, "nist")


if __name__ == "__main__":
    unittest.main()
