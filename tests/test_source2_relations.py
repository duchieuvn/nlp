from pathlib import Path
import sys
import unittest


RELATION_DIR = Path(__file__).resolve().parents[1] / "source2/relation"
sys.path.insert(0, str(RELATION_DIR))

from relation_builder import build_paper_relations
from relation_validation import validate_relation_payload


def paragraph(paragraph_id, text, sentence_id):
    return {
        "paragraph_id": paragraph_id,
        "text": text,
        "sentences": [{
            "sentence_id": sentence_id,
            "text": text,
        }],
    }


def fixture():
    document = {
        "paper_id": "paper",
        "sections": [
            {
                "section_id": "S1",
                "paragraphs": [
                    paragraph("S1.p1", "The shared model uses x and y.", "S1.p1.s1"),
                    paragraph(
                        "S1.p2",
                        "By substituting Eq. (1), we obtain the next expression.",
                        "S1.p2.s1",
                    ),
                ],
            },
            {
                "section_id": "S2",
                "paragraphs": [
                    paragraph(
                        "S2.p1",
                        "In the special case q=0, Equation (1) is recovered.",
                        "S2.p1.s1",
                    ),
                ],
            },
        ],
        "equations": [
            {
                "equation_id": "1",
                "section_id": "S1",
                "previous_paragraph_id": "S1.p1",
                "next_paragraph_id": None,
            },
            {
                "equation_id": "2",
                "section_id": "S1",
                "previous_paragraph_id": "S1.p2",
                "next_paragraph_id": None,
            },
            {
                "equation_id": "3",
                "section_id": "S2",
                "previous_paragraph_id": "S2.p1",
                "next_paragraph_id": None,
            },
        ],
        "cross_references": [],
    }
    symbols = {
        "paper_id": "paper",
        "equations": [
            {
                "equation_id": "1",
                "symbols": [{"canonical": "x"}, {"canonical": "y"}],
            },
            {
                "equation_id": "2",
                "symbols": [{"canonical": "x"}, {"canonical": "y"}],
            },
            {
                "equation_id": "3",
                "symbols": [{"canonical": "q"}],
            },
        ],
    }
    return document, symbols


class RelationTests(unittest.TestCase):
    def test_classifies_directional_derivation(self):
        payload = build_paper_relations(*fixture())

        relation = payload["equations"][1]["relations"]["1"]

        self.assertEqual(relation["grade"], "strong")
        self.assertEqual(relation["description"], "derived from")
        self.assertEqual(relation["source_sentence_id"], "S1.p2.s1")
        self.assertTrue(relation["audit"]["explicit_reference"])

    def test_shared_symbols_create_potential_reverse_edge(self):
        payload = build_paper_relations(*fixture())

        relation = payload["equations"][0]["relations"]["2"]

        self.assertEqual(relation["grade"], "potential")
        self.assertEqual(relation["description"], "shares symbols")
        self.assertEqual(relation["shared_symbols"], ["x", "y"])

    def test_classifies_special_case_and_none_edges(self):
        payload = build_paper_relations(*fixture())

        special = payload["equations"][2]["relations"]["1"]
        unrelated = payload["equations"][0]["relations"]["3"]

        self.assertEqual(special["grade"], "strong")
        self.assertEqual(special["description"], "special case")
        self.assertEqual(unrelated["grade"], "none")
        self.assertEqual(unrelated["description"], "")

    def test_emits_every_directed_pair(self):
        payload = build_paper_relations(*fixture())

        self.assertEqual(
            sum(len(equation["relations"]) for equation in payload["equations"]),
            6,
        )
        validate_relation_payload(payload)


if __name__ == "__main__":
    unittest.main()
