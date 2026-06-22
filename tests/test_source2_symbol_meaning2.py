import unittest

from source2.symbol_meaning2.html_index import build_inline_math_index
from source2.symbol_meaning2.selection import select_relation
from source2.symbol_meaning2.symbol_extractor import extract_symbol_meaning
from source2.symbol_meaning2.symbol_models import RelationPrediction
from source2.symbol_meaning2.symbol_parser import parse_symbol
from source2.symbol_meaning2.weak_supervision import split_by_paper


class SymbolMeaning2Tests(unittest.TestCase):
	@staticmethod
	def result(text):
		return {
			"rank": 1, "chunk_id": "paper:sentence:s1", "score": 10.0,
			"method": "bm25", "chunk_type": "sentence", "text": text,
			"paper_id": "paper", "section_title": "Model",
			"paragraph_ids": ["p1"], "sentence_ids": ["s1"],
			"nearby_equation_ids": ["1"],
		}
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

	def test_unreviewed_calibration_disables_neural_acceptance(self):
		parsed = parse_symbol({"canonical": "u", "latex_forms": ["u"]})
		prediction = RelationPrediction("displacement", "DEFINES_COMPLETE_SYMBOL", {"DEFINES_COMPLETE_SYMBOL": 1.0}, 1.0)
		selected, reason = select_relation([prediction], parsed, {
			"threshold": .9, "margin": .1, "acceptance_enabled": False,
		})
		self.assertIsNone(selected)
		self.assertEqual(reason, "reviewed_precision_gate_not_satisfied")

	def test_split_keeps_papers_together(self):
		examples = [{"paper_id": paper, "id": index} for paper in ("a", "b", "c") for index in range(3)]
		splits = split_by_paper(examples)
		locations = {row["paper_id"]: name for name, rows in splits.items() for row in rows}
		for paper in ("a", "b", "c"):
			self.assertEqual(sum(row["paper_id"] == paper for row in splits[locations[paper]]), 3)

	def test_regex_precedes_cross_encoder(self):
		class Classifier:
			def predict(self, parsed, candidates):
				raise AssertionError("classifier must not run for regex success")

		record = extract_symbol_meaning(
			{"canonical": "omega", "latex_forms": [r"\omega"], "aliases": [r"\omega"]},
			"1", [self.result(r"Here, \omega denotes angular frequency.")],
			classifier=Classifier(),
		)
		self.assertEqual(record.definition, "angular frequency")
		self.assertEqual(record.audit["selection_method"], "regex_precedence")

	def test_cross_encoder_accepts_extractable_complete_definition(self):
		class Classifier:
			model_name = "checkpoint"
			calibration = {"threshold": .9, "margin": .1, "acceptance_enabled": True}

			def predict(self, parsed, candidates):
				return [{"DEFINES_COMPLETE_SYMBOL": .96, "NO_RELATION": .04} for _ in candidates]

		record = extract_symbol_meaning(
			{"canonical": "omega", "latex_forms": [r"\omega"], "aliases": [r"\omega"]},
			"1", [self.result(r"We use the angular frequency \omega throughout.")],
			equation=r"E=\hbar\omega", classifier=Classifier(),
		)
		self.assertEqual(record.definition, "angular frequency")
		self.assertEqual(record.strategy, "mathbert_cross_encoder")
		self.assertTrue(record.audit["extractive"])

	def test_modifier_relation_is_audited_but_not_emitted(self):
		class Classifier:
			model_name = "checkpoint"
			calibration = {"threshold": .9, "margin": .1, "acceptance_enabled": True}

			def predict(self, parsed, candidates):
				return [{"QUALIFIES_SUPERSCRIPT": .97, "NO_RELATION": .03} for _ in candidates]

		record = extract_symbol_meaning(
			{"canonical": "u_n_sup_R", "latex_forms": [r"u^{R}_{n}"], "aliases": [r"u^{R}_{n}"]},
			"1", [self.result(r"The right displacement u^{R}_{n} appears in the solution.")],
			classifier=Classifier(),
		)
		self.assertEqual(record.definition, "")
		self.assertEqual(record.audit["component_relations"][0]["relation"], "QUALIFIES_SUPERSCRIPT")


if __name__ == "__main__":
	unittest.main()
