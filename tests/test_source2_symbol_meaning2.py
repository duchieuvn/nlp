import unittest

from source2.symbol_meaning2.html_index import build_inline_math_index
from source2.symbol_meaning2.selection import select_relation
from source2.symbol_meaning2.symbol_models import RelationPrediction
from source2.symbol_meaning2.symbol_parser import parse_symbol
from source2.symbol_meaning2.weak_supervision import split_by_paper


class SymbolMeaning2Tests(unittest.TestCase):
	def test_preserves_original_latex_and_parses_components(self):
		parsed = parse_symbol({"canonical": "u_n", "latex_forms": [r"u^{L}_{n}"], "aliases": ["u_n"]}, "u^L_n=x")
		self.assertEqual(parsed.original_latex, r"u^{L}_{n}")
		self.assertEqual((parsed.base, parsed.subscript, parsed.superscript), ("u", "n", "L"))

	def test_canonical_html_matching_preserves_latex(self):
		index = build_inline_math_index('<p id="p1">The <math id="m" alttext="u^{L}_{n}" display="inline"><mi>u</mi></math> is left.</p>')
		self.assertEqual(index["u^L_n"][0].latex, r"u^{L}_{n}")

	def test_component_only_relation_abstains(self):
		parsed = parse_symbol({"canonical": "u_n", "latex_forms": [r"u^{L}_{n}"]})
		prediction = RelationPrediction("left chain", "QUALIFIES_SUPERSCRIPT", {"QUALIFIES_SUPERSCRIPT": .95, "NO_RELATION": .05}, .95)
		selected, reason = select_relation([prediction], parsed)
		self.assertIsNone(selected)
		self.assertIsNotNone(reason)

	def test_low_margin_abstains(self):
		parsed = parse_symbol({"canonical": "u", "latex_forms": ["u"]})
		prediction = RelationPrediction("displacement", "DEFINES_COMPLETE_SYMBOL", {"DEFINES_COMPLETE_SYMBOL": .92, "DEFINES_BASE": .87}, .92)
		self.assertIsNone(select_relation([prediction], parsed)[0])

	def test_split_keeps_papers_together(self):
		examples = [{"paper_id": paper, "id": index} for paper in ("a", "b", "c") for index in range(3)]
		splits = split_by_paper(examples)
		locations = {row["paper_id"]: name for name, rows in splits.items() for row in rows}
		for paper in ("a", "b", "c"):
			self.assertEqual(sum(row["paper_id"] == paper for row in splits[locations[paper]]), 3)


if __name__ == "__main__":
	unittest.main()
