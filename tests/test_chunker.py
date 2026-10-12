import unittest
from src.precedentiq.data.legal_chunker import LegalSectionChunker
from src.precedentiq.schemas.judgment import LegalSectionEnum


class TestLegalSectionChunker(unittest.TestCase):
    def test_chunking_with_headings(self):
        judgment_text = """
        IN THE HIGH COURT OF MADHYA PRADESH
        WRIT PETITION No. 1218 of 2024
        
        Facts of the Case:
        The petitioner has approached this Court challenging the order dated 12.04.2023.
        The petitioner was duly appointed as Panchayat Karmi.
        
        Submissions of Counsel:
        Learned counsel for the petitioner submits that subsequent amendment cannot have retrospective effect.
        Per contra, State counsel vehemently opposes the contention.
        
        Ratio Decidendi:
        It is well settled law that rules prevailing at the time of advertisement govern the selection process.
        
        Order:
        Accordingly, the petition is allowed and the impugned order is hereby set aside.
        """

        chunks = LegalSectionChunker.chunk_judgment(
            text=judgment_text,
            judgment_id="TEST_CASE_001",
            court="high_court"
        )

        self.assertGreater(len(chunks), 0)
        sections = [c.section for c in chunks]
        self.assertIn(LegalSectionEnum.RATIO_DECIDENDI.value, sections)
        self.assertIn(LegalSectionEnum.ORDER.value, sections)

    def test_chunking_fallback(self):
        judgment_text = "Simple unstructured judgment text with no explicit section markers."
        chunks = LegalSectionChunker.chunk_judgment(
            text=judgment_text,
            judgment_id="TEST_CASE_002",
            court="supreme_court"
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].judgment_id, "TEST_CASE_002")


if __name__ == "__main__":
    unittest.main()
