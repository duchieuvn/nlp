import json
from pathlib import Path
import tempfile
import unittest

from analysis.export_meaning_postprocessing_report import build_report_data
from analysis.export_meaning_postprocessing_report import build_rejected_meanings
from analysis.export_meaning_postprocessing_report import export_report
from analysis.export_meaning_postprocessing_report import export_rejected_meanings


class MeaningPostprocessingReportTests(unittest.TestCase):
    def test_aggregates_overlapping_rejections_and_exports_markdown(self):
        payload = {
            "paper_id": "paper",
            "equations": [
                {
                    "equation_id": "1",
                    "meaning": "Hamiltonian",
                    "audit": {"postprocessing": {
                        "applied": True,
                        "strategy": "science_head_window",
                        "candidates": [],
                    }},
                },
                {
                    "equation_id": "2",
                    "meaning": "where x is a variable",
                    "source_text": "where x is a variable",
                    "audit": {"postprocessing": {
                        "applied": False,
                        "flagged": True,
                        "strategy": "no_reliable_phrase",
                        "original_meaning": "where x is a variable",
                        "candidates": [{
                            "reasons": [
                                "symbol_definition_sentence",
                                "missing_science_head",
                            ]
                        }],
                    }},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "paper.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            output_file = root / "report.md"
            rejected_file = root / "rejected.json"

            report = build_report_data(input_dir)
            export_report(input_dir, output_file)
            rejected = build_rejected_meanings(input_dir)
            export_rejected_meanings(input_dir, rejected_file)
            markdown = output_file.read_text(encoding="utf-8")
            rejected_json = json.loads(rejected_file.read_text(encoding="utf-8"))

        self.assertEqual(report["summary"]["records"], 2)
        self.assertEqual(report["summary"]["empty"], 0)
        self.assertEqual(report["summary"]["flagged"], 1)
        self.assertEqual(
            report["rejection_record_counts"]["symbol_definition_sentence"],
            1,
        )
        self.assertIn("# Equation Meaning Postprocessing Report", markdown)
        self.assertIn("where x is a variable", markdown)
        self.assertEqual(rejected["summary"]["rejected_count"], 1)
        rejected_record = rejected_json["rejected_meanings"][0]
        self.assertEqual(rejected_record["paper_id"], "paper")
        self.assertEqual(rejected_record["equation_id"], "2")
        self.assertEqual(
            rejected_record["original_meaning"], "where x is a variable"
        )
        self.assertEqual(
            rejected_record["rejection_reasons"],
            ["missing_science_head", "symbol_definition_sentence"],
        )


if __name__ == "__main__":
    unittest.main()
