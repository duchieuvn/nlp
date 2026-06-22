from collections import Counter, defaultdict
import json

from .config import (
	DEFAULT_RELATION_MARGIN,
	LABELS,
	MINIMUM_CALIBRATION_ACCEPTS,
	TARGET_ACCEPTED_PRECISION,
)


DEFINITION_LABELS = {"DEFINES_COMPLETE_SYMBOL", "DEFINES_BASE"}


def classification_metrics(predictions) -> dict:
	counts = defaultdict(lambda: Counter(tp=0, fp=0, fn=0, support=0))
	correct = 0
	confusion = {gold: {predicted: 0 for predicted in LABELS} for gold in LABELS}
	for prediction in predictions:
		gold = prediction.gold_label
		predicted = prediction.predicted_label
		counts[gold]["support"] += 1
		confusion[gold][predicted] += 1
		if gold == predicted:
			correct += 1
			counts[gold]["tp"] += 1
		else:
			counts[predicted]["fp"] += 1
			counts[gold]["fn"] += 1
	per_label = {}
	for label in LABELS:
		item = counts[label]
		precision = item["tp"] / max(1, item["tp"] + item["fp"])
		recall = item["tp"] / max(1, item["tp"] + item["fn"])
		f1 = 2 * precision * recall / max(1e-12, precision + recall)
		per_label[label] = {
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"support": item["support"],
		}
	active = [item for item in per_label.values() if item["support"]]
	return {
		"example_count": len(predictions),
		"accuracy": correct / max(1, len(predictions)),
		"macro_f1": sum(item["f1"] for item in active) / max(1, len(active)),
		"per_label": per_label,
		"confusion_matrix": confusion,
	}


def _accepts(prediction, threshold: float, margin: float) -> bool:
	if prediction.predicted_label not in DEFINITION_LABELS:
		return False
	if prediction.predicted_label == "DEFINES_BASE" and prediction.has_modifiers:
		return False
	ordered = sorted(prediction.probabilities.values(), reverse=True)
	winner = ordered[0]
	runner_up = ordered[1] if len(ordered) > 1 else 0.0
	return winner >= threshold and winner - runner_up >= margin


def calibrate_threshold(predictions) -> dict:
	scores = sorted({
		max(prediction.probabilities.values())
		for prediction in predictions
		if prediction.predicted_label in DEFINITION_LABELS
	})
	best = None
	for threshold in scores:
		accepted = [
			prediction for prediction in predictions
			if _accepts(prediction, threshold, DEFAULT_RELATION_MARGIN)
		]
		if len(accepted) < MINIMUM_CALIBRATION_ACCEPTS:
			continue
		correct = sum(
			prediction.gold_label == prediction.predicted_label
			for prediction in accepted
		)
		precision = correct / len(accepted)
		if precision >= TARGET_ACCEPTED_PRECISION:
			candidate = {
				"threshold": threshold,
				"margin": DEFAULT_RELATION_MARGIN,
				"accepted": len(accepted),
				"precision": precision,
				"coverage": len(accepted) / max(1, len(predictions)),
			}
			if best is None or candidate["coverage"] > best["coverage"]:
				best = candidate
	if best is None:
		best = {
			"threshold": 1.0,
			"margin": DEFAULT_RELATION_MARGIN,
			"accepted": 0,
			"precision": 0.0,
			"coverage": 0.0,
			"calibration_failed": True,
		}
	else:
		best["calibration_failed"] = False
	return best


def accepted_metrics(predictions, calibration: dict) -> dict:
	accepted = [
		prediction for prediction in predictions
		if _accepts(
			prediction, calibration["threshold"], calibration["margin"]
		)
	]
	correct = sum(
		prediction.gold_label == prediction.predicted_label
		for prediction in accepted
	)
	return {
		"accepted": len(accepted),
		"precision": correct / max(1, len(accepted)),
		"coverage": len(accepted) / max(1, len(predictions)),
		"remaining_abstention_rate": 1 - len(accepted) / max(1, len(predictions)),
	}


def build_evaluation(validation_predictions, test_predictions, history) -> dict:
	calibration = calibrate_threshold(validation_predictions)
	return {
		"evaluation_scope": "held-out weak labels; not human-reviewed ground truth",
		"training_history": history,
		"calibration": calibration,
		"validation": {
			"classification": classification_metrics(validation_predictions),
			"accepted_definitions": accepted_metrics(validation_predictions, calibration),
		},
		"test": {
			"classification": classification_metrics(test_predictions),
			"accepted_definitions": accepted_metrics(test_predictions, calibration),
		},
	}


def write_json(payload: dict, path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_report(dataset_summary: dict, evaluation: dict) -> str:
	test = evaluation["test"]
	accepted = test["accepted_definitions"]
	lines = [
		"# Symbol Relation Cross-Encoder Performance",
		"",
		"> Results use held-out weak labels, not human-reviewed ground truth. "
		"They measure bootstrap consistency and must not be presented as final "
		"scientific accuracy.",
		"",
		"## Dataset",
		"",
		f"- Examples: {dataset_summary['example_count']}",
		f"- Papers: {dataset_summary['paper_count']}",
		f"- Test papers: {dataset_summary['splits']['test']['papers']}",
		"- Split policy: paper-level deterministic split",
		"",
		"## Test Performance",
		"",
		f"- Accuracy: {test['classification']['accuracy']:.4f}",
		f"- Macro F1: {test['classification']['macro_f1']:.4f}",
		f"- Accepted-definition precision: {accepted['precision']:.4f}",
		f"- Accepted-definition coverage: {accepted['coverage']:.4f}",
		f"- Abstention rate: {accepted['remaining_abstention_rate']:.4f}",
		"",
		"## Calibration",
		"",
		f"- Probability threshold: {evaluation['calibration']['threshold']:.6f}",
		f"- Competing-label margin: {evaluation['calibration']['margin']:.2f}",
		f"- Target validation precision: {TARGET_ACCEPTED_PRECISION:.2f}",
		f"- Calibration failed: {evaluation['calibration']['calibration_failed']}",
		"",
		"## Per-Relation Test Metrics",
		"",
		"| Relation | Precision | Recall | F1 | Support |",
		"| --- | ---: | ---: | ---: | ---: |",
	]
	for label in LABELS:
		item = test["classification"]["per_label"][label]
		lines.append(
			f"| `{label}` | {item['precision']:.4f} | {item['recall']:.4f} "
			f"| {item['f1']:.4f} | {item['support']} |"
		)
	lines.extend((
		"",
		"## Limitations",
		"",
		"- Labels are bootstrapped from regex decisions and BM25 rejection evidence.",
		"- Rare modifier relations may remain underrepresented.",
		"- A manually reviewed benchmark is required before production promotion.",
		"",
	))
	return "\n".join(lines)
