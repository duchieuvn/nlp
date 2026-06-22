import unittest

from source2.symbol_meaning_finetune.data import RelationExample, split_by_paper
from source2.symbol_meaning_finetune.evaluation import (
	accepted_metrics,
	calibrate_threshold,
	classification_metrics,
	render_report,
)
from source2.symbol_meaning_finetune.model import (
	EncodedRelationDataset,
	Prediction,
	select_device,
)


def example(paper_id: str, label="NO_RELATION") -> RelationExample:
	return RelationExample(
		paper_id, "1", "x", "x", "x=1", "value", "sentence",
		label, "test", False,
	)


class FineTuningPipelineTests(unittest.TestCase):
	def test_split_has_no_paper_leakage(self):
		examples = [example(str(paper)) for paper in range(20) for _ in range(2)]
		splits = split_by_paper(examples)
		papers = {name: {row.paper_id for row in rows} for name, rows in splits.items()}
		self.assertFalse(papers["train"] & papers["validation"])
		self.assertFalse(papers["train"] & papers["test"])
		self.assertFalse(papers["validation"] & papers["test"])
		self.assertTrue(all(splits.values()))

	def test_encoded_dataset_collates_as_tensors(self):
		import torch
		from torch.utils.data import DataLoader

		class Tokenizer:
			def __call__(self, left, right, **kwargs):
				return {
					"input_ids": [[1, 2, 0] for _ in left],
					"attention_mask": [[1, 1, 0] for _ in left],
				}

		dataset = EncodedRelationDataset([example("a"), example("b")], Tokenizer())
		batch = next(iter(DataLoader(dataset, batch_size=2)))
		self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
		self.assertTrue(all(isinstance(value, torch.Tensor) for value in batch.values()))

	def test_device_selection_falls_back_to_cpu(self):
		class Cuda:
			@staticmethod
			def is_available():
				return True

			@staticmethod
			def synchronize():
				raise RuntimeError("device busy")

			@staticmethod
			def get_device_name(device):
				return "test"

		class Torch:
			AcceleratorError = RuntimeError
			cuda = Cuda()

			@staticmethod
			def empty(*args, **kwargs):
				return object()

			@staticmethod
			def device(name):
				return name

		self.assertEqual(select_device(Torch()), "cpu")

	def test_calibration_enforces_precision_and_margin(self):
		predictions = [
			Prediction(
				"DEFINES_COMPLETE_SYMBOL", "DEFINES_COMPLETE_SYMBOL",
				{"DEFINES_COMPLETE_SYMBOL": .96, "NO_RELATION": .04},
				False, "p", "1", "x", "value",
			)
			for _ in range(5)
		] + [Prediction(
			"NO_RELATION", "DEFINES_COMPLETE_SYMBOL",
			{"DEFINES_COMPLETE_SYMBOL": .60, "NO_RELATION": .40},
			False, "q", "1", "y", "wrong",
		)]
		calibration = calibrate_threshold(predictions)
		self.assertFalse(calibration["calibration_failed"])
		self.assertGreater(calibration["threshold"], .60)
		self.assertEqual(accepted_metrics(predictions, calibration)["precision"], 1.0)

	def test_metrics_and_report_are_explicitly_weak_labelled(self):
		prediction = Prediction(
			"NO_RELATION", "NO_RELATION", {"NO_RELATION": 1.0},
			False, "p", "1", "x", "none",
		)
		metrics = classification_metrics([prediction])
		self.assertEqual(metrics["accuracy"], 1.0)
		summary = {
			"example_count": 1,
			"paper_count": 1,
			"splits": {"test": {"papers": 1}},
		}
		evaluation = {
			"test": {
				"classification": metrics,
				"accepted_definitions": {
					"precision": 0.0, "coverage": 0.0,
					"remaining_abstention_rate": 1.0,
				},
			},
			"calibration": {
				"threshold": 1.0, "margin": .1,
				"calibration_failed": True,
			},
		}
		self.assertIn("weak labels", render_report(summary, evaluation))


if __name__ == "__main__":
	unittest.main()
