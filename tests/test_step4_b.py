from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

import step4_b


class CharacterTokenizer:
	model_max_length = 512

	def encode(self, text, add_special_tokens=False):
		return [ord(character) for character in text]

	def decode(self, token_ids, **kwargs):
		return "".join(chr(token_id) for token_id in token_ids)

	def num_special_tokens_to_add(self, pair=False):
		return 3 if pair else 2


class FakeQaPipeline:
	def __init__(self, answer, score=0.9, offset_shift=0):
		self.tokenizer = CharacterTokenizer()
		self.answer = answer
		self.score = score
		self.offset_shift = offset_shift

	def __call__(self, *, question, context):
		start = context.index(self.answer) + self.offset_shift
		return {
			"answer": self.answer,
			"score": self.score,
			"start": start,
			"end": start + len(self.answer),
		}


class Step4BTests(unittest.TestCase):
	def test_model_window_keeps_marker_within_pair_limit(self):
		tokenizer = CharacterTokenizer()
		window = f"{'x' * 100} {step4_b.EQUATION_MARKER} {'y' * 100}"

		result = step4_b.model_input_window(window, tokenizer, 100)

		self.assertIn(step4_b.EQUATION_MARKER, result)
		pair_length = (
			len(tokenizer.encode(step4_b.QA_QUESTION))
			+ len(tokenizer.encode(result))
			+ tokenizer.num_special_tokens_to_add(pair=True)
		)
		self.assertLessEqual(pair_length, 100)

	def test_predicts_and_cleans_exact_source_span(self):
		window = "We derive the Markovian master equation [EQUATION]."
		pipeline = FakeQaPipeline("the Markovian master equation")

		result = step4_b.predict_meaning(window, pipeline)

		self.assertEqual(result["meaning"], "Markovian master equation")
		self.assertEqual(result["raw_answer"], "the Markovian master equation")
		self.assertEqual(result["status"], "accepted")
		self.assertEqual(
			result["source_text"][result["start"] : result["end"]],
			result["raw_answer"],
		)

	def test_rejects_low_confidence_answer(self):
		pipeline = FakeQaPipeline("master equation", score=0.49)

		result = step4_b.predict_meaning(
			"The master equation reads [EQUATION].",
			pipeline,
		)

		self.assertEqual(result["meaning"], "")
		self.assertEqual(result["status"], "low_confidence")

	def test_rejects_answer_with_invalid_offsets(self):
		pipeline = FakeQaPipeline("master equation", offset_shift=1)

		result = step4_b.predict_meaning(
			"The master equation reads [EQUATION].",
			pipeline,
		)

		self.assertEqual(result["status"], "invalid_span")

	def test_rejects_cross_reference(self):
		pipeline = FakeQaPipeline("Eq. 12")

		result = step4_b.predict_meaning(
			"Using Eq. 12 gives [EQUATION].",
			pipeline,
		)

		self.assertEqual(result["status"], "rejected_candidate")

	def test_extract_meanings_clears_stale_meaning_and_appends_audit(self):
		data = {
			"paper": {
				"1": {
					"meaning": "stale value",
					"surrounding_text": {"window": ""},
					"audit-trail": [],
				}
			}
		}
		with tempfile.TemporaryDirectory() as directory:
			input_file = Path(directory) / "input.json"
			output_file = Path(directory) / "output.json"
			input_file.write_text(json.dumps(data), encoding="utf-8")
			with patch.object(step4_b, "load_qa_pipeline", return_value=FakeQaPipeline("unused")):
				counts = step4_b.extract_meanings(input_file, output_file)

			output = json.loads(output_file.read_text(encoding="utf-8"))

		self.assertEqual(counts, (1, 0, 1))
		entry = output["paper"]["1"]
		self.assertEqual(entry["meaning"], "")
		audit = entry["audit-trail"][-1]["meaning_extraction"]
		self.assertEqual(audit["status"], "missing_context")
		self.assertEqual(audit["strategy"], "extractive_qa")


if __name__ == "__main__":
	unittest.main()
