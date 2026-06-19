from __future__ import annotations

from pathlib import Path
import sys
import unittest


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "source" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import equation_ner


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeNerPipeline:
    tokenizer = CharacterTokenizer()

    def __init__(self, entities):
        self.entities = entities

    def __call__(self, source_text):
        return self.entities


class EquationNerTests(unittest.TestCase):
    def test_formula_context_uses_token_bounded_preview(self):
        result = equation_ner.format_model_context(
            "A name precedes [EQUATION] after.",
            "abcdefghij",
            CharacterTokenizer(),
            context_mode="formula",
            max_equation_tokens=4,
        )

        self.assertEqual(result, "A name precedes [EQUATION] abcd after.")

    def test_formula_context_keeps_marker_for_empty_or_malformed_equation(self):
        window = "Before [EQUATION] after"

        self.assertEqual(
            equation_ner.format_model_context(
                window, "", CharacterTokenizer(), context_mode="formula"
            ),
            window,
        )

    def test_source_span_reconstruction_uses_offsets_not_subwords(self):
        text = "Wigner characteristic function [EQUATION]"
        entities = [
            {"entity": "B-EQ_NAME", "word": "Wig", "start": 0, "end": 6, "score": 0.9},
            {"entity": "I-EQ_NAME", "word": "##ner", "start": 6, "end": 7, "score": 0.8},
            {"entity": "I-EQ_NAME", "word": "characteristic", "start": 7, "end": 22, "score": 0.9},
            {"entity": "I-EQ_NAME", "word": "function", "start": 22, "end": 30, "score": 0.9},
        ]

        result = equation_ner.predict_meaning(text, FakeNerPipeline(entities))

        self.assertEqual(result["meaning"], "Wigner characteristic function")
        self.assertEqual((result["start"], result["end"]), (0, 30))
        self.assertNotIn("##", result["meaning"])

    def test_paper_split_has_no_cross_split_leakage(self):
        examples = [
            equation_ner.NerExample(paper, str(index), ["x"], ["O"], "", "x", has_answer=False)
            for paper in ("a", "a", "b", "c", "d", "e")
            for index in range(2)
        ]

        splits = equation_ner.split_examples(examples, 0.6, 0.2, seed=7)
        paper_sets = [{example.paper_id for example in split} for split in splits.values()]

        self.assertFalse(paper_sets[0] & paper_sets[1])
        self.assertFalse(paper_sets[0] & paper_sets[2])
        self.assertFalse(paper_sets[1] & paper_sets[2])

    def test_review_validation_accepts_exact_positive_and_negative(self):
        positive = {
            "review_status": "reviewed",
            "has_answer": True,
            "window": "the master equation reads [EQUATION]",
            "meaning": "master equation",
            "start": 4,
            "end": 19,
        }
        negative = {
            "review_status": "reviewed",
            "has_answer": False,
            "window": "we obtain [EQUATION]",
            "meaning": "",
            "start": None,
            "end": None,
        }

        self.assertEqual(equation_ner.validate_reviewed_record(positive), "")
        self.assertEqual(equation_ner.validate_reviewed_record(negative), "")

    def test_metrics_include_answer_and_no_answer_behavior(self):
        gold = [
            {"paper_id": "p", "equation_id": "1", "has_answer": True, "meaning": "master equation"},
            {"paper_id": "p", "equation_id": "2", "has_answer": False, "meaning": ""},
        ]
        predictions = {
            ("p", "1"): {"meaning": "master equation", "status": "accepted"},
            ("p", "2"): {"meaning": "", "status": "no_answer"},
        }

        metrics = equation_ner.evaluate_predictions(gold, predictions)

        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["answer_precision"], 1.0)
        self.assertEqual(metrics["answer_recall"], 1.0)
        self.assertEqual(metrics["no_answer_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
