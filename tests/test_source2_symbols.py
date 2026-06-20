import json
from pathlib import Path
import sys
import tempfile
import unittest


SYMBOL_DIR = Path(__file__).resolve().parents[1] / "source2/symbol"
sys.path.insert(0, str(SYMBOL_DIR))

from chunk_enricher import enrich_chunk_file
from symbol_builder import build_paper_symbols
from symbol_extractor import extract_symbols


class SymbolExtractionTests(unittest.TestCase):
    def test_extracts_normalized_symbols_and_aliases(self):
        symbols = {
            symbol.canonical: symbol
            for symbol in extract_symbols(
                r"H_{Lambda}=\omega_{c}a^{\dagger}a+\hat{Q}+i+j+\text{noise}"
            )
        }

        self.assertIn("H_Lambda", symbols)
        self.assertIn("omega_c", symbols)
        self.assertIn("a", symbols)
        self.assertIn("Q_hat", symbols)
        self.assertNotIn("i", symbols)
        self.assertNotIn("j", symbols)
        self.assertNotIn("e", symbols)
        self.assertNotIn("noise", symbols)
        self.assertIn("ω_c", symbols["omega_c"].aliases)
        self.assertIn(r"\omega_{c}", symbols["omega_c"].latex_forms)

    def test_preserves_prime_and_nested_subscript_names(self):
        names = {
            symbol.canonical
            for symbol in extract_symbols(r"F_{1}^{\prime}+V_{A_{1}A_{2}}")
        }

        self.assertIn("F_1_prime", names)
        self.assertIn("V_A_1_A_2", names)

    def test_preserves_decorators_subscripts_and_symbolic_superscripts(self):
        names = {
            symbol.canonical
            for symbol in extract_symbols(r"\ddot{u}_{n}+q^{L}+x^2")
        }

        self.assertIn("u_n_double_dot", names)
        self.assertIn("q_sup_L", names)
        self.assertIn("x", names)

    def test_builds_registry_and_enriches_nearby_chunks(self):
        registry = build_paper_symbols("paper", {
            "1": {"equation": r"H=\omega_c a"},
            "2": {"equation": r"E=mc^2"},
        })
        with tempfile.TemporaryDirectory() as temporary_dir:
            chunk_file = Path(temporary_dir) / "paper.json"
            chunk_file.write_text(json.dumps({
                "paper_id": "paper",
                "chunks": [{
                    "chunk_id": "paper:sentence:s1",
                    "nearby_equation_ids": ["1"],
                    "symbols": [],
                }],
            }), encoding="utf-8")

            enriched = enrich_chunk_file(chunk_file, registry)
            chunk = json.loads(chunk_file.read_text())["chunks"][0]

            self.assertEqual(enriched, 1)
            self.assertEqual(chunk["symbols"], ["H", "omega_c", "a"])


if __name__ == "__main__":
    unittest.main()
