import json
from pathlib import Path
import sys
import tempfile
import unittest


CHUNK_BUILDER_DIR = Path(__file__).resolve().parents[1] / "source2/chunk_builder"
sys.path.insert(0, str(CHUNK_BUILDER_DIR))

from builder import build_chunk_views
from chunk_io import build_chunk_file


def fixture_paper():
    sentence_1 = {
        "sentence_id": "S1.p1.s1",
        "order": 1,
        "text": "Equation (1) defines the state.",
        "start": 0,
        "end": 31,
        "cross_reference_ids": ["paper:xref:1"],
    }
    sentence_2 = {
        "sentence_id": "S1.p2.s1",
        "order": 1,
        "text": "The state evolves in time.",
        "start": 0,
        "end": 26,
        "cross_reference_ids": [],
    }
    return {
        "paper_id": "paper",
        "title": "Fixture Paper",
        "html_source": "fixture.html",
        "source_status": "html",
        "sections": [{
            "section_id": "S1",
            "parent_section_id": None,
            "order": 1,
            "level": 1,
            "kind": "section",
            "title": "Model",
            "synthetic": False,
            "paragraphs": [
                {
                    "paragraph_id": "S1.p1",
                    "order": 1,
                    "document_order": 1,
                    "text": sentence_1["text"],
                    "sentences": [sentence_1],
                    "nearby_equation_ids": ["1"],
                    "cross_reference_ids": ["paper:xref:1"],
                },
                {
                    "paragraph_id": "S1.p2",
                    "order": 2,
                    "document_order": 3,
                    "text": sentence_2["text"],
                    "sentences": [sentence_2],
                    "nearby_equation_ids": ["1"],
                    "cross_reference_ids": [],
                },
            ],
            "equation_ids": ["1"],
        }],
        "equations": [{
            "equation_id": "1",
            "latex": "x=y",
            "section_id": "S1",
            "document_order": 2,
            "anchor_id": "S1.E1",
            "annotation_ids": [],
            "match_method": "visible_label",
            "previous_paragraph_id": "S1.p1",
            "next_paragraph_id": "S1.p2",
            "legacy_context_before": "",
            "legacy_context_after": "",
        }],
        "cross_references": [{
            "reference_id": "paper:xref:1",
            "raw_text": "Equation (1)",
            "reference_type": "singular",
            "source_section_id": "S1",
            "source_paragraph_id": "S1.p1",
            "source_sentence_id": "S1.p1.s1",
            "paragraph_start": 0,
            "paragraph_end": 12,
            "sentence_start": 0,
            "sentence_end": 12,
            "target_equation_ids": ["1"],
            "unresolved_labels": [],
        }],
    }


class ChunkBuilderTests(unittest.TestCase):
    def test_builds_all_chunk_views_with_source_metadata(self):
        chunks = build_chunk_views(fixture_paper())
        by_type = {}
        for chunk in chunks:
            by_type.setdefault(chunk["chunk_type"], []).append(chunk)

        self.assertEqual(
            set(by_type),
            {
                "sentence",
                "paragraph",
                "equation_neighborhood",
                "section_aware",
                "cross_reference",
            },
        )
        self.assertEqual(len(by_type["sentence"]), 2)
        self.assertEqual(len(by_type["paragraph"]), 2)
        self.assertEqual(len(by_type["equation_neighborhood"]), 1)
        self.assertEqual(len(by_type["section_aware"]), 2)
        self.assertEqual(len(by_type["cross_reference"]), 1)

        neighborhood = by_type["equation_neighborhood"][0]
        self.assertIn("Section: Model", neighborhood["text"])
        self.assertIn("Equation (1): x=y", neighborhood["text"])
        self.assertEqual(neighborhood["paragraph_ids"], ["S1.p1", "S1.p2"])
        self.assertEqual(neighborhood["nearby_equation_ids"], ["1"])
        self.assertEqual(neighborhood["symbols"], [])

    def test_writes_one_chunk_file_per_paper(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_file = root / "paper.json"
            input_file.write_text(json.dumps(fixture_paper()), encoding="utf-8")

            paper_id, chunk_count = build_chunk_file(input_file, root / "chunks")
            payload = json.loads((root / "chunks/paper.json").read_text())

            self.assertEqual(paper_id, "paper")
            self.assertEqual(chunk_count, 8)
            self.assertEqual(len(payload["chunks"]), 8)
            self.assertTrue(all(chunk["paper_id"] == "paper" for chunk in payload["chunks"]))


if __name__ == "__main__":
    unittest.main()
