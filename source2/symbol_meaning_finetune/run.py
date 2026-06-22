from pathlib import Path
import sys


if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source2.symbol_meaning_finetune.config import (
	CHECKPOINT_DIR,
	DATASET_SUMMARY_FILE,
	INFERENCE_CONFIG_FILE,
	METRICS_FILE,
	REJECTED_EVIDENCE_FILE,
	REPORT_FILE,
	SYMBOL_MEANINGS_DIR,
	SYMBOLS_DIR,
)
from source2.symbol_meaning_finetune.data import (
	build_examples,
	dataset_summary,
	split_by_paper,
)
from source2.symbol_meaning_finetune.evaluation import (
	build_evaluation,
	render_report,
	write_json,
)
from source2.symbol_meaning_finetune.model import predict, train_model


def _validate_inputs() -> None:
	missing = [
		path for path in (SYMBOLS_DIR, SYMBOL_MEANINGS_DIR, REJECTED_EVIDENCE_FILE)
		if not path.exists()
	]
	if missing:
		raise FileNotFoundError(
			"Missing fine-tuning inputs: " + ", ".join(map(str, missing))
		)


def main() -> None:
	_validate_inputs()
	print("Stage 1/4: building weak relation examples")
	examples = build_examples(
		SYMBOLS_DIR, SYMBOL_MEANINGS_DIR, REJECTED_EVIDENCE_FILE
	)
	splits = split_by_paper(examples)
	if not all(splits.values()):
		raise RuntimeError("Paper-level split produced an empty dataset partition")
	summary = dataset_summary(examples, splits)
	write_json(summary, DATASET_SUMMARY_FILE)
	print(
		f"Built {summary['example_count']} examples from "
		f"{summary['paper_count']} papers"
	)

	print("Stage 2/4: fine-tuning MathBERT")
	model, tokenizer, device, history = train_model(splits, CHECKPOINT_DIR)

	print("Stage 3/4: calibrating on held-out validation papers")
	validation_predictions = predict(
		model, tokenizer, device, splits["validation"]
	)
	test_predictions = predict(model, tokenizer, device, splits["test"])
	evaluation = build_evaluation(
		validation_predictions, test_predictions, history
	)
	write_json(evaluation, METRICS_FILE)
	write_json({
		"relation_threshold": evaluation["calibration"]["threshold"],
		"relation_margin": evaluation["calibration"]["margin"],
		"labels": list(model.config.label2id),
		"calibration_scope": evaluation["evaluation_scope"],
	}, INFERENCE_CONFIG_FILE)

	print("Stage 4/4: writing performance report")
	REPORT_FILE.write_text(
		render_report(summary, evaluation), encoding="utf-8"
	)
	print(f"Checkpoint: {CHECKPOINT_DIR}")
	print(f"Performance report: {REPORT_FILE}")


if __name__ == "__main__":
	main()
