import unittest
from src.precedentiq.schemas.judgment import JudgmentChunk, ChunkMetadata
from src.precedentiq.retrieval.bm25_retriever import LegalTokenizer, BM25Retriever
from src.precedentiq.schemas.retrieval import FilterParams


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.chunk1 = JudgmentChunk(
            chunk_id="chunk_1",
            judgment_id="CASE_SC_01",
            citation="AIR 2022 SC 100",
            court="supreme_court",
            section="order",
            text="Whether power of attorney holder can institute legal proceedings on behalf of the principal.",
            metadata=ChunkMetadata(source_year="2022")
        )
        self.chunk2 = JudgmentChunk(
            chunk_id="chunk_2",
            judgment_id="CASE_HC_02",
            citation="2023 MPHC 500",
            court="high_court",
            section="order",
            text="The petition seeking quashing of FIR under Section 482 CrPC is allowed as dispute is civil.",
            metadata=ChunkMetadata(source_year="2023")
        )
        self.retriever = BM25Retriever([self.chunk1, self.chunk2])

    def test_legal_tokenizer(self):
        tokens = LegalTokenizer.tokenize("Section 138 of the Negotiable Instruments Act and Article 226")
        self.assertIn("sec_138", tokens)
        self.assertIn("art_226", tokens)

    def test_bm25_search_hit(self):
        results = self.retriever.search("power of attorney principal", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score, rank = results[0]
        self.assertEqual(top_chunk.chunk_id, "chunk_1")

    def test_court_filter(self):
        filt = FilterParams(court="supreme_court")
        results = self.retriever.search("proceedings", top_k=2, filters=filt)
        for chunk, _, _ in results:
            self.assertEqual(chunk.court, "supreme_court")


if __name__ == "__main__":
    unittest.main()
