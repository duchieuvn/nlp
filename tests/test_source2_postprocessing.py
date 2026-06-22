import json
from pathlib import Path
import sys
import tempfile
import unittest


POSTPROCESSING_DIR = Path(__file__).resolve().parents[1] / "source2/postprocessing"
sys.path.insert(0, str(POSTPROCESSING_DIR))

from meaning_cleaner import FORBIDDEN_VERBS, MAX_MEANING_WORDS, WORD
from meaning_cleaner import clean_meaning, postprocess_record
from postprocessing_io import postprocess_directory, postprocess_file
from postprocessing_io import summarize_directory


class MeaningPostprocessingTests(unittest.TestCase):
    def test_extracts_tmst_subject_phrase(self):
        text = (
            "Since TMST state is a Gaussian state with zero mean and covariance "
            "matrix given by Eq. ( 1 ), the Wigner characteristic function of "
            "the TMST state can be readily evaluated using Eq. ( 25 ):"
        )

        result = clean_meaning(text)

        self.assertEqual(
            result.meaning,
            "Wigner characteristic function of the TMST state",
        )
        self.assertEqual(result.strategy, "subject_before_introduction")

    def test_extracts_subject_before_passive_introduction(self):
        result = clean_meaning(
            "The covariance matrix of the TMSV state can be written as"
        )

        self.assertEqual(result.meaning, "covariance matrix of the TMSV state")

    def test_extracts_context_object(self):
        result = clean_meaning(
            "We simplify this expression to derive the dispersion relation."
        )

        self.assertEqual(result.meaning, "dispersion relation")
        self.assertEqual(result.strategy, "derived_object")

    def test_extracts_named_complement(self):
        result = clean_meaning(
            "In some circles, the time interval is called the correlation hole."
        )

        self.assertEqual(result.meaning, "correlation hole")
        self.assertEqual(result.strategy, "named_complement")

    def test_rejects_symbol_definition_sentence(self):
        result = clean_meaning(
            "where p is the momentum of the particle and m is its mass."
        )

        self.assertEqual(result.meaning, "")
        self.assertEqual(result.strategy, "no_reliable_phrase")
        self.assertIn(
            "symbol_definition_sentence",
            result.candidates[0]["reasons"],
        )

    def test_preserves_qualified_latex(self):
        result = clean_meaning(
            "The unitary evolution operator of \\hat{H}_{2} can be represented as"
        )

        self.assertEqual(
            result.meaning,
            "unitary evolution operator of \\hat{H}_{2}",
        )

    def test_rejects_bare_math(self):
        self.assertEqual(clean_meaning("\\rho=\\sum_i p_i").meaning, "")

    def test_extracts_science_head_from_procedural_sentence(self):
        text = (
            "Taking into account the plasma dispersion relation, we expand "
            "the wave equations to first order."
        )

        self.assertEqual(clean_meaning(text).meaning, "plasma dispersion relation")

    def test_cleans_trusted_existing_phrase(self):
        result = clean_meaning(
            "the dispersion relation",
            source_strategy="derived_object",
        )

        self.assertEqual(result.meaning, "dispersion relation")
        self.assertEqual(result.strategy, "existing_phrase")

    def test_nonempty_result_obeys_phrase_invariants(self):
        source = (
            "The effective system Hamiltonian can be represented using a matrix."
        )
        result = clean_meaning(source)
        words = [word.casefold() for word in WORD.findall(result.meaning)]

        self.assertTrue(result.meaning in source)
        self.assertLessEqual(len(words), MAX_MEANING_WORDS)
        self.assertFalse(set(words) & FORBIDDEN_VERBS)

    def test_adds_candidate_audit_metadata(self):
        source = "The effective Hamiltonian can be written as"
        record = postprocess_record({
            "meaning": source,
            "source_text": source,
            "strategy": "source_sentence",
            "audit": {"candidate_count": 2},
        })

        self.assertEqual(record["meaning"], "effective Hamiltonian")
        postprocessing = record["audit"]["postprocessing"]
        self.assertTrue(postprocessing["applied"])
        self.assertTrue(postprocessing["extractive"])
        self.assertTrue(postprocessing["candidates"])
        self.assertEqual(record["audit"]["candidate_count"], 2)

    def test_preserves_meaning_when_no_candidate_passes_validation(self):
        source = "where p is the momentum of the particle and m is its mass."

        record = postprocess_record({
            "meaning": source,
            "source_text": source,
            "strategy": "source_sentence",
        })

        self.assertEqual(record["meaning"], source)
        postprocessing = record["audit"]["postprocessing"]
        self.assertFalse(postprocessing["applied"])
        self.assertTrue(postprocessing["flagged"])
        self.assertEqual(postprocessing["strategy"], "no_reliable_phrase")

    def test_record_postprocessing_is_idempotent(self):
        source = "The effective Hamiltonian can be written as"
        record = {
            "meaning": source,
            "source_text": source,
            "strategy": "source_sentence",
        }

        first = postprocess_record(record)
        second = postprocess_record(first)

        self.assertEqual(second, first)

    def test_postprocesses_json_atomically(self):
        source = "The effective Hamiltonian can be written as"
        payload = {
            "paper_id": "paper",
            "equations": [{
                "meaning": source,
                "source_text": source,
                "strategy": "source_sentence",
            }],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_file = root / "input.json"
            output_file = root / "output.json"
            input_file.write_text(json.dumps(payload), encoding="utf-8")

            records, changed = postprocess_file(input_file, output_file)
            output = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual((records, changed), (1, 1))
        self.assertEqual(
            output["equations"][0]["meaning"],
            "effective Hamiltonian",
        )

    def test_directory_processing_does_not_overwrite_inputs(self):
        source = "The effective Hamiltonian can be written as"
        payload = {
            "paper_id": "paper",
            "equations": [{
                "meaning": source,
                "source_text": source,
                "strategy": "source_sentence",
            }],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            input_file = input_dir / "paper.json"
            original = json.dumps(payload)
            input_file.write_text(original, encoding="utf-8")

            counts = postprocess_directory(input_dir, output_dir)

            self.assertEqual(input_file.read_text(encoding="utf-8"), original)
            self.assertTrue((output_dir / "paper.json").exists())
            self.assertEqual(counts, (1, 1, 1))

    def test_rejects_same_input_and_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            with self.assertRaisesRegex(ValueError, "must differ"):
                postprocess_directory(directory, directory)

    def test_summarizes_phrase_results(self):
        source = "The effective Hamiltonian can be written as"
        payload = {
            "paper_id": "paper",
            "equations": [{
                "meaning": source,
                "source_text": source,
                "strategy": "source_sentence",
            }],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_file = input_dir / "paper.json"
            write_file.write_text(json.dumps(payload), encoding="utf-8")
            postprocess_directory(input_dir, output_dir)

            summary = summarize_directory(output_dir)

        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["nonempty"], 1)
        self.assertEqual(summary["validation_failures"], 0)
        self.assertEqual(summary["flagged"], 0)


if __name__ == "__main__":
    unittest.main()
