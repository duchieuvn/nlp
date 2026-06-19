from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

import s4b


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]


class FakeTorch:
    @staticmethod
    def dot(left, right):
        return left * right


class MathBertBaselineTests(unittest.TestCase):
    def test_extracts_literal_equation_name_candidates(self):
        window = (
            "The covariance matrix of the thermal state can be written as "
            "[EQUATION] where x is fixed."
        )

        candidates = s4b.candidate_phrases(window)

        texts = {candidate.text for candidate in candidates}
        self.assertIn("covariance matrix of the thermal state", texts)
        for candidate in candidates:
            self.assertEqual(window[candidate.start : candidate.end], candidate.text)

    def test_context_keeps_marker_and_nearby_text(self):
        window = f"{'old ' * 500}important name {s4b.EQUATION_MARKER} after context"

        result = s4b.model_context(window)

        self.assertIn(s4b.EQUATION_MARKER, result)
        self.assertIn("important name", result)
        self.assertIn("after context", result)

    def test_returns_blank_when_no_natural_language_candidate_exists(self):
        result = s4b.predict_meaning(
            "123 [EQUATION] 456", "x=y", None, None, None, "cpu"
        )

        self.assertEqual(result["meaning"], "")
        self.assertEqual(result["status"], "no_candidate")

    def test_prediction_returns_exact_source_offsets(self):
        window = "The master equation is given by [EQUATION]."
        with patch.object(s4b, "embed_texts", return_value=FakeTensor([1.0] * 50)):
            result = s4b.predict_meaning(
                window, "x=y", object(), object(), FakeTorch(), "cpu"
            )

        self.assertEqual(window[result["start"] : result["end"]], result["candidate"])
        self.assertEqual(result["status"], "accepted")

    def test_run_uses_fixed_equation_input_and_writes_compatible_output(self):
        data = {
            "paper": {
                "1": {
                    "equation": "x=y",
                    "meaning": "",
                    "surrounding_text": {"window": "master equation [EQUATION]"},
                    "audit-trail": [],
                }
            }
        }
        prediction = {
            "meaning": "master equation",
            "candidate": "master equation",
            "confidence": 0.8,
            "status": "accepted",
            "start": 0,
            "end": 15,
        }
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "3_equations.json"
            output_file = Path(directory) / "4b_mathbert_baseline.json"
            input_file.write_text(json.dumps(data), encoding="utf-8")
            with (
                patch.object(s4b, "INPUT_FILE", input_file),
                patch.object(s4b, "OUTPUT_FILE", output_file),
                patch.object(s4b, "load_mathbert", return_value=(None, None, None, "cpu")),
                patch.object(s4b, "predict_meaning", return_value=prediction),
            ):
                counts = s4b.run_baseline()
            output = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(counts, (1, 1))
        entry = output["paper"]["1"]
        self.assertEqual(entry["meaning"], "master equation")
        audit = entry["audit-trail"][-1]["meaning_extraction"]
        self.assertEqual(audit["model"], s4b.MODEL_NAME)
        self.assertEqual(audit["strategy"], "mathberta_embedding_baseline")


if __name__ == "__main__":
    unittest.main()
