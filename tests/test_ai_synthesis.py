import unittest
from src.precedentiq.schemas.judgment import JudgmentChunk, ChunkMetadata
from src.precedentiq.ai.fallbacks import HeuristicLegalSynthesizer


class TestAISynthesis(unittest.TestCase):
    def setUp(self):
        self.chunk1 = JudgmentChunk(
            chunk_id="chk_01",
            judgment_id="MPHC_2023_01",
            citation="MPHC_2023_01",
            court="high_court",
            section="order",
            text="It is well settled law that a power of attorney holder cannot depose for the principal on acts of personal knowledge. Accordingly, the petition is allowed with liberty.",
            metadata=ChunkMetadata(treatment_signals=["followed"])
        )
        self.chunk2 = JudgmentChunk(
            chunk_id="chk_02",
            judgment_id="MPHC_2023_02",
            citation="MPHC_2023_02",
            court="high_court",
            section="order",
            text="The petition stands dismissed for non-appearance and failure to submit the relevant registered power of attorney.",
            metadata=ChunkMetadata(treatment_signals=["reversed"])
        )

    def test_holding_extraction(self):
        holding = HeuristicLegalSynthesizer.extract_holding([self.chunk1], "MPHC_2023_01")
        self.assertIsNotNone(holding.ratio_decidendi)
        self.assertIn("power of attorney", holding.ratio_decidendi.lower())
        self.assertIsNotNone(holding.final_order)

    def test_argument_synthesis(self):
        outline = HeuristicLegalSynthesizer.synthesize_arguments(
            query="Power of attorney holder locus",
            chunks=[self.chunk1, self.chunk2]
        )
        self.assertGreater(len(outline.supporting_arguments), 0)
        self.assertGreater(len(outline.counter_arguments), 0)

        # Check citation links
        for arg in outline.supporting_arguments:
            self.assertGreater(len(arg.supporting_citations), 0)
            self.assertIn(arg.supporting_citations[0].chunk_id, ["chk_01", "chk_02"])


if __name__ == "__main__":
    unittest.main()
