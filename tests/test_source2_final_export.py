import json
from pathlib import Path
import tempfile
import unittest

from source2.final_export.exporter import build_final_data, export_final_data


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FinalExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.equations_file = self.root / "equations.json"
        self.paper_list_file = self.root / "papers.txt"
        self.meanings_dir = self.root / "meanings"
        self.symbols_dir = self.root / "symbols"
        self.relations_dir = self.root / "relations"
        write_json(self.equations_file, {
            "empty-paper": {},
            "paper": {
                "1": {"equation": "a=b", "audit-trail": ["ignored"]},
                "2": {"equation": "b=c", "audit-trail": ["ignored"]},
            },
        })
        self.paper_list_file.write_text(
            "arXiv:paper\narXiv:unrelated\narXiv:empty-paper\n",
            encoding="utf-8",
        )
        write_json(self.meanings_dir / "paper.json", {
            "paper_id": "paper",
            "equations": [
                {
                    "equation_id": "1",
                    "equation": "a=b",
                    "meaning": "postprocessed meaning",
                    "audit": {"ignored": True},
                },
                {
                    "equation_id": "2",
                    "equation": "b=c",
                    "meaning": "",
                    "audit": {"ignored": True},
                },
            ],
        })
        write_json(self.symbols_dir / "paper.json", {
            "paper_id": "paper",
            "equations": [
                {
                    "equation_id": "1",
                    "latex": "a=b",
                    "symbols": [
                        {"canonical": "a", "definition": "first quantity", "aliases": ["a"]},
                        {"canonical": "b", "definition": "", "aliases": ["b"]},
                    ],
                },
                {"equation_id": "2", "latex": "b=c", "symbols": []},
            ],
        })
        write_json(self.relations_dir / "paper.json", {
            "paper_id": "paper",
            "equations": [
                {
                    "equation_id": "1",
                    "relations": {
                        "2": {
                            "grade": "strong",
                            "description": "derived from",
                            "score": 9.0,
                            "audit": {"ignored": True},
                        }
                    },
                },
                {
                    "equation_id": "2",
                    "relations": {
                        "1": {
                            "grade": "none",
                            "description": "",
                            "score": 0.0,
                            "audit": {"ignored": True},
                        }
                    },
                },
            ],
        })

    def tearDown(self):
        self.temporary_dir.cleanup()

    def build(self):
        return build_final_data(
            self.equations_file,
            self.paper_list_file,
            self.meanings_dir,
            self.symbols_dir,
            self.relations_dir,
        )

    def test_builds_strict_schema_and_uses_postprocessed_meaning(self):
        payload = self.build()

        self.assertEqual(list(payload), ["paper", "empty-paper"])
        self.assertEqual(payload["empty-paper"], {})
        entry = payload["paper"]["1"]
        self.assertEqual(
            list(entry), ["equation", "meaning", "symbols", "relations"]
        )
        self.assertEqual(entry["meaning"], "postprocessed meaning")
        self.assertEqual(
            entry["symbols"], {"a": "first quantity", "b": ""}
        )
        self.assertEqual(
            entry["relations"],
            {"2": {"grade": "strong", "description": "derived from"}},
        )
        self.assertNotIn("audit", json.dumps(entry))
        self.assertNotIn("aliases", json.dumps(entry))

    def test_preserves_none_relations_and_empty_symbol_objects(self):
        entry = self.build()["paper"]["2"]

        self.assertEqual(entry["symbols"], {})
        self.assertEqual(
            entry["relations"],
            {"1": {"grade": "none", "description": ""}},
        )

    def test_rejects_duplicate_canonical_symbols(self):
        path = self.symbols_dir / "paper.json"
        payload = json.loads(path.read_text())
        payload["equations"][0]["symbols"].append({
            "canonical": "a", "definition": "duplicate"
        })
        write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "Duplicate canonical symbol"):
            self.build()

    def test_rejects_missing_component_equation(self):
        path = self.meanings_dir / "paper.json"
        payload = json.loads(path.read_text())
        payload["equations"].pop()
        write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "equation mismatch"):
            self.build()

    def test_rejects_equation_text_mismatch(self):
        path = self.symbols_dir / "paper.json"
        payload = json.loads(path.read_text())
        payload["equations"][0]["latex"] = "wrong"
        write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "Symbol equation mismatch"):
            self.build()

    def test_rejects_base_paper_missing_from_order_file(self):
        self.paper_list_file.write_text("arXiv:paper\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Papers missing from paper list"):
            self.build()

    def test_failure_does_not_replace_existing_output(self):
        output_file = self.root / "final.json"
        output_file.write_text("previous", encoding="utf-8")
        (self.meanings_dir / "paper.json").unlink()

        with self.assertRaises(ValueError):
            export_final_data(
                self.equations_file,
                self.paper_list_file,
                self.meanings_dir,
                self.symbols_dir,
                self.relations_dir,
                output_file,
            )

        self.assertEqual(output_file.read_text(encoding="utf-8"), "previous")


if __name__ == "__main__":
    unittest.main()
