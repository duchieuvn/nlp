from .symbol_config import DEFAULT_RELATION_MARGIN, DEFAULT_RELATION_THRESHOLD
from .symbol_models import ParsedSymbol, RelationPrediction


def select_relation(
	predictions: list[RelationPrediction],
	parsed: ParsedSymbol,
	calibration: dict | None = None,
) -> tuple[RelationPrediction | None, str | None]:
	calibration = calibration or {
		"threshold": DEFAULT_RELATION_THRESHOLD,
		"margin": DEFAULT_RELATION_MARGIN,
		"acceptance_enabled": True,
	}
	if not calibration.get("acceptance_enabled", True):
		return None, "reviewed_precision_gate_not_satisfied"
	eligible = []
	reasons = []
	for prediction in predictions:
		ordered = sorted(prediction.probabilities.values(), reverse=True)
		winning = prediction.probabilities.get(
			prediction.relation, prediction.cross_encoder_score
		)
		competing = ordered[1] if len(ordered) > 1 else 0.0
		if winning < calibration["threshold"]:
			reasons.append("below_relation_threshold")
			continue
		if winning - competing < calibration["margin"]:
			reasons.append("ambiguous_relation_margin")
			continue
		if prediction.relation == "DEFINES_COMPLETE_SYMBOL":
			eligible.append(prediction)
		elif prediction.relation == "DEFINES_BASE" and not parsed.has_semantic_modifiers:
			eligible.append(prediction)
		else:
			reasons.append("component_or_non_definition_relation")
	if not eligible:
		return None, reasons[0] if reasons else "no_definition_relation"
	return max(
		eligible,
		key=lambda prediction: (
			prediction.cross_encoder_score,
			-(prediction.candidate.result.get("rank", 0) if prediction.candidate else 0),
			prediction.phrase,
		),
	), None
