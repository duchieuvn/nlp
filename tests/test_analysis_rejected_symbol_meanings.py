import json
from pathlib import Path
import tempfile
import unittest

from analysis.export_rejected_symbol_meanings import build_rejected_symbol_meanings
from analysis.export_rejected_symbol_meanings import render_markdown


class Result:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class Service:
    def search(self, query, method):
        return [Result({
            "chunk_id": "paper:sentence:1",
            "chunk_type": "sentence",
            "rank": 1,
            "score": 4.0,
            "method": method,
            "text": "The value x occurs here without a definition.",
            "paragraph_ids": [],
            "sentence_ids": [],
            "nearby_equation_ids": ["1"],
        })]


class RejectedSymbolMeaningTests(unittest.TestCase):
    def test_exports_only_empty_definitions_with_candidate_reasons(self):
        payload = {
            "paper_id": "paper",
            "equations": [{
                "equation_id": "1",
                "latex": "x+y",
                "symbols": [
                    {"canonical": "x", "latex_forms": ["x"], "aliases": ["x"], "definition": ""},
                    {"canonical": "y", "latex_forms": ["y"], "aliases": ["y"], "definition": "output"},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "paper.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = build_rejected_symbol_meanings(input_dir, Service())
            markdown = render_markdown(report)

        self.assertEqual(report["summary"]["symbol_count"], 2)
        self.assertEqual(report["summary"]["rejected_count"], 1)
        self.assertEqual(report["summary"]["empty_percentage"], 50.0)
        record = report["rejected_symbol_meanings"][0]
        self.assertEqual(record["canonical"], "x")
        self.assertEqual(
            record["rejection_reasons"],
            ["no_supported_definition_pattern"],
        )
        self.assertTrue(record["candidates"][0]["alias_mentioned"])
        self.assertIn("# Symbol Meaning Rejection Report", markdown)
        self.assertIn("| Empty definitions | 1 |", markdown)
        self.assertIn("`no_supported_definition_pattern`", markdown)
        self.assertIn("The value x occurs here", markdown)


if __name__ == "__main__":
    unittest.main()
