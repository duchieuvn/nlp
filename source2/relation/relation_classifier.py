from relation_config import POTENTIAL_THRESHOLD, STRONG_THRESHOLD
from relation_context import EquationContext, explicit_evidence, jaccard, token_set
from relation_patterns import (
	DERIVATION_CUE,
	EQUIVALENCE_CUE,
	SPECIAL_CASE_CUE,
	cue_near_equation,
	from_equation_cue,
)


def classify_relation(
	source: EquationContext,
	target: EquationContext,
	source_symbols: set[str],
	target_symbols: set[str],
	cross_references: list[dict],
	section_positions: dict[str, int],
) -> dict:
	evidence = explicit_evidence(
		source, target.equation["equation_id"], cross_references
	)
	target_id = target.equation["equation_id"]
	best_evidence = max(
		evidence,
		key=lambda item: (
			cue_near_equation(EQUIVALENCE_CUE, item.text, target_id),
			cue_near_equation(SPECIAL_CASE_CUE, item.text, target_id),
			cue_near_equation(DERIVATION_CUE, item.text, target_id)
			or from_equation_cue(item.text, target_id),
			-item.text.count(" "),
		),
		default=None,
	)
	evidence_text = best_evidence.text if best_evidence else ""
	explicit = bool(evidence)
	derivation = explicit and (
		cue_near_equation(DERIVATION_CUE, evidence_text, target_id)
		or from_equation_cue(evidence_text, target_id)
	)
	equivalence = explicit and cue_near_equation(EQUIVALENCE_CUE, evidence_text, target_id)
	special_case = explicit and cue_near_equation(SPECIAL_CASE_CUE, evidence_text, target_id)
	shared_symbols = sorted(source_symbols & target_symbols)
	symbol_union = source_symbols | target_symbols
	shared_jaccard = (
		len(shared_symbols) / len(symbol_union) if symbol_union else 0.0
	)
	same_section = source.equation.get("section_id") == target.equation.get("section_id")
	context_similarity = jaccard(token_set(source.text), token_set(target.text))
	source_section_position = section_positions.get(source.equation.get("section_id"), 0)
	target_section_position = section_positions.get(target.equation.get("section_id"), 0)
	section_distance = abs(source_section_position - target_section_position)

	features: dict[str, float] = {}
	if explicit:
		features["explicit_cross_reference"] = 5.0
	if derivation:
		features["derivation_cue"] = 4.0
	if equivalence:
		features["equivalence_cue"] = 3.0
	if special_case:
		features["special_case_cue"] = 3.0
	if shared_jaccard:
		features["shared_symbol_jaccard"] = round(2.0 * shared_jaccard, 4)
	if same_section:
		features["same_section"] = 1.0
	if context_similarity:
		features["context_similarity"] = round(context_similarity, 4)
	if section_distance and not explicit:
		features["section_distance_penalty"] = -float(min(2, section_distance))
	score = round(sum(features.values()), 4)

	if explicit or score >= STRONG_THRESHOLD:
		grade = "strong"
	elif score >= POTENTIAL_THRESHOLD:
		grade = "potential"
	else:
		grade = "none"

	if grade == "none":
		description = ""
	elif equivalence:
		description = "equivalent"
	elif special_case:
		description = "special case"
	elif derivation:
		description = "derived from"
	elif explicit:
		description = "explicit citation"
	elif shared_symbols:
		description = "shares symbols"
	else:
		description = "same section context"

	return {
		"grade": grade,
		"description": description,
		"score": score,
		"shared_symbols": shared_symbols,
		"evidence_text": evidence_text,
		"source_sentence_id": best_evidence.sentence_id if best_evidence else None,
		"source_sentence_ids": list(best_evidence.sentence_ids) if best_evidence else [],
		"source_paragraph_id": best_evidence.paragraph_id if best_evidence else None,
		"audit": {
			"features": features,
			"explicit_reference": explicit,
			"derivation_cue": derivation,
			"equivalence_cue": equivalence,
			"special_case_cue": special_case,
			"shared_symbol_jaccard": round(shared_jaccard, 4),
			"context_similarity": round(context_similarity, 4),
			"same_section": same_section,
			"section_distance": section_distance,
			"strong_threshold": STRONG_THRESHOLD,
			"potential_threshold": POTENTIAL_THRESHOLD,
		},
	}
