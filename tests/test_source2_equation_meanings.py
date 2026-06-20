import json
from pathlib import Path
import sys
import tempfile
import unittest


MEANING_DIR = Path(__file__).resolve().parents[1] / "source2/eqn_meaning"
sys.path.insert(0, str(MEANING_DIR))

from meaning_extractor import extract_equation_meaning
from meaning_io import extract_all_meanings


def result(chunk_id, text, score=10.0, rank=1):
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "score": score,
        "method": "bm25",
        "chunk_type": "equation_neighborhood",
        "text": text,
        "paper_id": "paper",
        "section_id": "S1",
        "section_title": "Model",
        "paragraph_ids": ["S1.p1"],
        "sentence_ids": ["S1.p1.s1"],
        "nearby_equation_ids": ["1"],
        "symbols": ["H"],
        "source": "html",
    }


class EquationMeaningTests(unittest.TestCase):
    class FakeReranker:
        model_name = "test/mathbert"

        def score_candidates(self, equation, query, candidates):
            return [0.1 if "Hamiltonian" in item.text else 0.9 for item in candidates]

    def test_extracts_subject_before_equation(self):
        query = {
            "equation_id": "1",
            "results": [result(
                "paper:equation_neighborhood:1",
                "Therefore, the effective system Hamiltonian can be written as\n"
                "Equation (1): H=H_0+V\n"
                "where V is the interaction.",
            )],
        }

        record = extract_equation_meaning(query, "H=H_0+V")

        self.assertEqual(record.meaning, "the effective system Hamiltonian")
        self.assertEqual(record.strategy, "anchor_subject")
        self.assertTrue(record.audit["extractive"])

    def test_extracts_derived_concept(self):
        query = {
            "equation_id": "3",
            "results": [result(
                "paper:equation_neighborhood:3",
                "Equation (3): omega^2=f(q)\n"
                "We simplify this expression to derive the dispersion relation.",
            )],
        }

        record = extract_equation_meaning(query, "omega^2=f(q)")

        self.assertEqual(record.meaning, "the dispersion relation")
        self.assertEqual(record.strategy, "derived_object")

    def test_returns_empty_for_weak_evidence(self):
        query = {
            "equation_id": "1",
            "results": [result(
                "paper:paragraph:p1",
                "The experiment was repeated yesterday.",
            )],
        }

        record = extract_equation_meaning(query, "x=y")

        self.assertEqual(record.meaning, "")
        self.assertEqual(record.strategy, "no_reliable_candidate")

    def test_mathbert_similarity_reranks_eligible_candidates(self):
        query = {
            "equation_id": "1",
            "results": [
                result(
                    "paper:equation_neighborhood:1",
                    "The effective Hamiltonian is\nEquation (1): H=H_0+V",
                    score=10.0,
                    rank=1,
                ),
                result(
                    "paper:equation_neighborhood:1",
                    "The interaction energy is\nEquation (1): H=H_0+V",
                    score=9.0,
                    rank=2,
                ),
            ],
        }

        record = extract_equation_meaning(
            query, "H=H_0+V", reranker=self.FakeReranker()
        )

        self.assertEqual(record.meaning, "The interaction energy")
        self.assertEqual(record.audit["mathbert_model"], "test/mathbert")
        self.assertEqual(record.audit["mathbert_similarity"], 0.9)
        self.assertEqual(record.audit["mathbert_normalized_score"], 1.0)

    def test_mathbert_fallback_bypasses_minimum_score(self):
        query = {
            "equation_id": "1",
            "results": [result(
                "paper:paragraph:p1",
                "The experiment was repeated yesterday.",
            )],
        }

        record = extract_equation_meaning(
            query, "x=y", reranker=self.FakeReranker()
        )

        self.assertEqual(record.meaning, "The experiment was repeated yesterday.")
        self.assertEqual(record.audit["selection_method"], "mathbert_fallback")
        self.assertTrue(record.audit["minimum_score_bypassed"])
        self.assertLess(record.candidate_score, 6.0)

    def test_accepts_and_flags_procedural_fragments(self):
        query = {
            "equation_id": "2",
            "results": [result(
                "paper:equation_neighborhood:2",
                "Equation (2): x=y\nWe substitute Eq. (2) into Eq. (1) and obtain",
            )],
        }

        record = extract_equation_meaning(query, "x=y")

        self.assertEqual(
            record.meaning,
            "We substitute Eq. (2) into Eq. (1) and obtain",
        )
        self.assertTrue(record.audit["hard_filtered"])
        self.assertEqual(record.audit["hard_filtered_paper_id"], "paper")
        self.assertIn(
            "procedural_or_incomplete",
            record.audit["hard_filter_reasons"],
        )

    def test_extracts_object_from_write_down_anchor(self):
        query = {
            "equation_id": "1",
            "results": [result(
                "paper:equation_neighborhood:1",
                "We write down the motion equation in the form\n"
                "Equation (1): x=y",
            )],
        }

        record = extract_equation_meaning(query, "x=y")

        self.assertEqual(record.meaning, "the motion equation")
        self.assertEqual(record.strategy, "anchor_object")

    def test_extracts_subject_after_discourse_prefix(self):
        query = {
            "equation_id": "6",
            "results": [result(
                "paper:equation_neighborhood:6",
                "So the displacement of the n-th atom is\n"
                "Equation (6): u_n=Ae^{iqn}",
            )],
        }

        record = extract_equation_meaning(query, "u_n=Ae^{iqn}")

        self.assertEqual(record.meaning, "the displacement of the n-th atom")
        self.assertEqual(record.strategy, "anchor_subject")

    def test_exports_one_file_per_paper(self):
        retrieval = {
            "paper_id": "paper",
            "method": "bm25",
            "top_k": 10,
            "queries": [{
                "equation_id": "1",
                "results": [result(
                    "paper:equation_neighborhood:1",
                    "The effective Hamiltonian is\nEquation (1): H=H_0+V",
                )],
            }],
        }
        equations = {"paper": {"1": {"equation": "H=H_0+V"}}}
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            retrieval_dir = root / "retrieval"
            retrieval_dir.mkdir()
            (retrieval_dir / "paper.json").write_text(
                json.dumps(retrieval), encoding="utf-8"
            )
            equations_file = root / "equations.json"
            equations_file.write_text(json.dumps(equations), encoding="utf-8")

            counts = extract_all_meanings(
                retrieval_dir, equations_file, root / "meanings"
            )
            payload = json.loads((root / "meanings/paper.json").read_text())

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(payload["equations"][0]["meaning"], "The effective Hamiltonian")


if __name__ == "__main__":
    unittest.main()
