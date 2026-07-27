from __future__ import annotations

import unittest

from rag_starter.agent_retrieval import AgenticRetriever
from rag_starter.retrieval import BM25Retriever
from rag_starter.schema import Chunk


def make_chunk(
    source: str,
    text: str,
    *,
    status: str = "final",
    superseded_by: str | None = None,
    topics: tuple[str, ...] = (),
) -> Chunk:
    return Chunk(
        text=text,
        source=source,
        doc_id=source,
        chunk_id=f"{source}::1",
        start_char=0,
        end_char=len(text),
        title="TLS 1.3" if source.startswith("RFC") else "HMAC",
        status=status,
        superseded_by=superseded_by,
        topics=topics,
        section_id="1",
    )


class AgenticRetrieverTest(unittest.TestCase):
    def test_current_version_route_demotes_obsolete_document(self) -> None:
        chunks = [
            make_chunk(
                "RFC.8446",
                "TLS 1.3 cipher suite requirements",
                status="obsolete",
                superseded_by="RFC.9846",
            ),
            make_chunk("RFC.9846", "TLS 1.3 current cipher suite requirements"),
        ]
        retriever = AgenticRetriever(chunks, BM25Retriever(chunks), candidate_k=10)

        results = retriever.search("当前 TLS 1.3 应该采用 RFC 8446 还是 RFC 9846？", top_k=2)

        self.assertEqual(results[0].chunk.doc_id, "RFC.9846")
        self.assertTrue(retriever.last_plan["current_version"])

    def test_comparison_reserves_slots_for_named_sources(self) -> None:
        chunks = [
            make_chunk("RFC.2104", "HMAC ipad opad construction"),
            make_chunk("NIST.FIPS.198-1", "HMAC inner outer construction"),
            make_chunk("OTHER", "HMAC HMAC HMAC unrelated overview"),
        ]
        retriever = AgenticRetriever(
            chunks,
            BM25Retriever(chunks),
            candidate_k=10,
            max_per_source=1,
        )

        results = retriever.search(
            "对照 RFC 2104 与 FIPS 198-1 的 HMAC 结构",
            top_k=2,
        )

        self.assertEqual(
            {result.chunk.doc_id for result in results},
            {"RFC.2104", "NIST.FIPS.198-1"},
        )

    def test_topic_route_recovers_distinctive_algorithm_document(self) -> None:
        chunks = [
            make_chunk(
                "RFC.9106",
                "The salt length is 16 bytes.",
                topics=("argon2", "password-hashing"),
            ),
            make_chunk("OTHER", "salt salt salt password"),
        ]
        retriever = AgenticRetriever(chunks, BM25Retriever(chunks), candidate_k=10)

        results = retriever.search("Argon2 的 salt 长度是多少？", top_k=1)

        self.assertEqual(results[0].chunk.doc_id, "RFC.9106")
        self.assertEqual(retriever.last_plan["routed_sources"], ["RFC.9106"])

    def test_nist_revision_suffix_does_not_hide_explicit_source(self) -> None:
        chunks = [
            make_chunk("NIST.SP.800-108r1-upd1", "Label and Context"),
            make_chunk("OTHER", "Label and Context"),
        ]
        retriever = AgenticRetriever(chunks, BM25Retriever(chunks), candidate_k=10)

        retriever.search("SP 800-108 的 Label 和 Context 是什么？", top_k=1)

        self.assertEqual(
            retriever.last_plan["explicit_sources"],
            ["NIST.SP.800-108r1-upd1"],
        )

    def test_protocol_context_can_override_generic_algorithm_route(self) -> None:
        chunks = [
            make_chunk(
                "RFC.9106",
                "Argon2 salt memory-hard password hashing",
                topics=("argon2",),
            ),
            make_chunk(
                "RFC.9580",
                "OpenPGP Argon2 S2K descriptor salt passes parallelism memory",
                topics=("openpgp",),
            ),
        ]
        retriever = AgenticRetriever(chunks, BM25Retriever(chunks), candidate_k=10)

        results = retriever.search(
            "OpenPGP Argon2 S2K 如何编码 salt、passes 和 memory？",
            top_k=2,
        )

        self.assertEqual(results[0].chunk.doc_id, "RFC.9580")
        self.assertEqual(
            set(retriever.last_plan["routed_sources"]),
            {"RFC.9106", "RFC.9580"},
        )


if __name__ == "__main__":
    unittest.main()
