from pathlib import Path
import sys
import unittest
from unittest.mock import patch


MEANING_DIR = Path(__file__).resolve().parents[1] / "source2/symbol_meaning"
sys.path.insert(0, str(MEANING_DIR))

from symbol_extractor import extract_symbol_meaning
from symbol_io import extract_paper_symbol_meanings


def result(text, score=10.0, rank=1, equation_id="1"):
    return {
        "rank": rank,
        "chunk_id": "paper:sentence:s1",
        "score": score,
        "method": "bm25",
        "chunk_type": "sentence",
        "text": text,
        "paper_id": "paper",
        "section_id": "S1",
        "section_title": "Model",
        "paragraph_ids": ["S1.p1"],
        "sentence_ids": ["S1.p1.s1"],
        "nearby_equation_ids": [equation_id],
        "symbols": ["omega"],
        "source": "html",
    }


def symbol(canonical="omega", aliases=None):
    return {
        "canonical": canonical,
        "latex_forms": [r"\omega"],
        "aliases": aliases or [canonical, r"\omega", "ω"],
    }


class SymbolMeaningTests(unittest.TestCase):
    def test_extracts_definition_after_symbol(self):
        record = extract_symbol_meaning(
            symbol(),
            "1",
            [result(r"Here, \omega denotes the angular frequency of the mode.")],
        )

        self.assertEqual(record.definition, "angular frequency of the mode")
        self.assertEqual(record.strategy, "symbol_before_cue")
        self.assertEqual(record.audit["matched_alias"], r"\omega")
        self.assertTrue(record.audit["extractive"])

    def test_extracts_definition_before_symbol(self):
        target = symbol("u_n", ["u_n", r"u^{L}_{n}"])
        record = extract_symbol_meaning(
            target,
            "1",
            [result(
                r"We denote the displacement of the atom on the left side as u^{L}_{n}."
            )],
        )

        self.assertEqual(
            record.definition,
            "displacement of the atom on the left side",
        )
        self.assertEqual(record.strategy, "definition_before_symbol")

    def test_rejects_symbol_mentions_without_definition_pattern(self):
        record = extract_symbol_meaning(
            symbol(),
            "1",
            [result(r"We substitute \omega into the previous equation.")],
        )

        self.assertEqual(record.definition, "")
        self.assertEqual(record.strategy, "no_reliable_definition")

    def test_rejects_bare_copular_property(self):
        record = extract_symbol_meaning(
            symbol(),
            "1",
            [result(r"The expression shows that \omega is small.")],
        )

        self.assertEqual(record.definition, "")

    def test_accepts_copular_definition_in_where_clause(self):
        record = extract_symbol_meaning(
            symbol(),
            "1",
            [result(r"Here, \omega is the angular frequency of the mode.")],
        )

        self.assertEqual(record.definition, "angular frequency of the mode")
        self.assertEqual(record.strategy, "contextual_copula")

    def test_rejects_property_in_where_clause(self):
        record = extract_symbol_meaning(
            symbol("t", ["t"]),
            "1",
            [result("Here, time t is discretized into Trotter steps.")],
        )

        self.assertEqual(record.definition, "")

    def test_truncates_coordinated_symbol_definitions(self):
        record = extract_symbol_meaning(
            symbol("c", ["c"]),
            "1",
            [result(
                r"Here, c is the speed of light, and \varepsilon_0 is vacuum permittivity."
            )],
        )

        self.assertEqual(record.definition, "speed of light")

    def test_rejects_truncated_definition_span(self):
        record = extract_symbol_meaning(
            symbol("R", ["R"]),
            "1",
            [result("Here, R is a probability mass function p(r(t.")],
        )

        self.assertEqual(record.definition, "")

    @patch("symbol_io.retrieve_symbol_evidence")
    def test_builds_audited_paper_payload(self, retrieve):
        retrieve.return_value = (
            r"omega \omega denotes represents where",
            [result(r"The symbol \omega denotes angular frequency.")],
        )
        payload = extract_paper_symbol_meanings(object(), {
            "paper_id": "paper",
            "equations": [{
                "equation_id": "1",
                "latex": r"E=\hbar\omega",
                "symbols": [symbol()],
            }],
        })

        record = payload["equations"][0]["symbols"][0]
        self.assertEqual(record["definition"], "angular frequency")
        self.assertEqual(record["audit"]["query"], retrieve.return_value[0])
        self.assertEqual(payload["retrieval_method"], "bm25")


if __name__ == "__main__":
    unittest.main()
