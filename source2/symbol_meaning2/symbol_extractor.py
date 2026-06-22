from .candidates import collect_sentences, extract_phrase_candidates
from .patterns import extract_regex_definition, mentions_alias
from .selection import select_relation
from .symbol_config import MAX_NEURAL_CANDIDATES, MINIMUM_DEFINITION_SCORE
from .symbol_models import RelationPrediction, SymbolMeaningRecord
from .symbol_parser import parse_symbol


def _score_regex(result: dict, equation_id: str, maximum: float, strategy: str):
	features = {"alias_definition_pattern": 4.0, "definition_cue": 2.0}
	if equation_id in result.get("nearby_equation_ids", []):
		features["near_target_equation"] = 2.0
	if result.get("chunk_type") == "sentence":
		features["sentence_chunk"] = 1.0
	if strategy in {"definition_before_symbol", "let_be"}:
		features["explicit_definition_construction"] = 1.0
	if maximum > 0:
		features["retrieval_score"] = result.get("score", 0.0) / maximum
	return sum(features.values()), features


def _base_audit(parsed, sentences, classifier) -> dict:
	calibration = getattr(classifier, "calibration", None)
	return {
		"candidate_count": len(sentences),
		"minimum_definition_score": MINIMUM_DEFINITION_SCORE,
		"original_latex": parsed.original_latex,
		"parsed_components": parsed.to_dict(),
		"cross_encoder_model": getattr(classifier, "model_name", None),
		"calibration": calibration,
	}


def _empty_record(
	symbol, parsed, sentences, reason, classifier=None, component_relations=None,
) -> SymbolMeaningRecord:
	audit = _base_audit(parsed, sentences, classifier)
	audit.update({
		"rejection_reason": reason,
		"component_relations": component_relations or [],
	})
	return SymbolMeaningRecord(
		canonical=symbol["canonical"],
		latex_forms=list(symbol.get("latex_forms", [])),
		aliases=list(dict.fromkeys(symbol.get("aliases", []))),
		definition="",
		confidence=0.0,
		strategy="no_reliable_definition",
		source_text="",
		source_chunk_id=None,
		audit=audit,
	)


def _regex_record(symbol, parsed, equation_id, sentences, classifier=None):
	aliases = list(dict.fromkeys(symbol.get("aliases", [])))
	maximum = max((result.get("score", 0.0) for _, result in sentences), default=0.0)
	scored = []
	for text, result in sentences:
		if not mentions_alias(text, aliases):
			continue
		definition, strategy, matched_alias = extract_regex_definition(text, aliases)
		if not definition:
			continue
		score, features = _score_regex(result, equation_id, maximum, strategy)
		scored.append((score, text, result, definition, strategy, matched_alias, features))
	if not scored:
		return None
	winner = max(scored, key=lambda item: (item[0], -item[2].get("rank", 0), item[3]))
	if winner[0] < MINIMUM_DEFINITION_SCORE:
		return None
	score, text, result, definition, strategy, matched_alias, features = winner
	audit = _base_audit(parsed, sentences, classifier)
	audit.update({
		"matched_alias": matched_alias,
		"score_features": features,
		"selection_method": "regex_precedence",
		"extractive": definition in text,
		"rejection_reason": None,
		"component_relations": [],
	})
	return SymbolMeaningRecord(
		canonical=symbol["canonical"],
		latex_forms=list(symbol.get("latex_forms", [])),
		aliases=aliases,
		definition=definition,
		confidence=round(min(1.0, score / 11.0), 4),
		strategy=strategy,
		source_text=text,
		source_chunk_id=result.get("chunk_id"),
		source_paragraph_ids=list(result.get("paragraph_ids", [])),
		source_sentence_ids=list(result.get("sentence_ids", [])),
		retrieval_method=result.get("method"),
		retrieval_rank=result.get("rank"),
		retrieval_score=result.get("score"),
		candidate_score=round(score, 4),
		audit=audit,
	)


def extract_symbol_meaning(
	symbol: dict,
	equation_id: str,
	results: list[dict],
	equation: str = "",
	nlp=None,
	classifier=None,
) -> SymbolMeaningRecord:
	parsed = parse_symbol(symbol, equation)
	sentences = collect_sentences(results)
	regex_record = _regex_record(
		symbol, parsed, equation_id, sentences, classifier
	)
	if regex_record is not None:
		return regex_record
	if classifier is None:
		reason = (
			"no_supported_definition_pattern"
			if any(mentions_alias(text, parsed.aliases) for text, _ in sentences)
			else "no_retrieved_alias"
		)
		return _empty_record(symbol, parsed, sentences, reason)

	candidates = extract_phrase_candidates(sentences, parsed.aliases, nlp)
	candidates = sorted(
		candidates,
		key=lambda candidate: (
			candidate.result.get("rank", 10**9),
			-candidate.result.get("score", 0.0),
			candidate.phrase,
		),
	)[:MAX_NEURAL_CANDIDATES]
	if not candidates:
		return _empty_record(
			symbol, parsed, sentences, "no_extractive_phrase_candidates", classifier
		)
	probability_rows = classifier.predict(parsed, candidates)
	if len(probability_rows) != len(candidates):
		raise ValueError("Cross-encoder returned an invalid number of predictions")
	predictions = []
	for candidate, probabilities in zip(candidates, probability_rows):
		relation, score = max(probabilities.items(), key=lambda item: item[1])
		predictions.append(RelationPrediction(
			candidate.phrase, relation, probabilities, score, candidate
		))
	winner, rejection_reason = select_relation(
		predictions, parsed, getattr(classifier, "calibration", None)
	)
	relation_audit = [{
		"candidate_phrase": prediction.phrase,
		"evidence_sentence": prediction.candidate.sentence,
		"candidate_source": prediction.candidate.source,
		"relation": prediction.relation,
		"relation_probabilities": prediction.probabilities,
		"cross_encoder_score": prediction.cross_encoder_score,
		"source_chunk_id": prediction.candidate.result.get("chunk_id"),
		"source_paragraph_ids": list(prediction.candidate.result.get("paragraph_ids", [])),
		"source_sentence_ids": list(prediction.candidate.result.get("sentence_ids", [])),
	} for prediction in predictions]
	if winner is None:
		return _empty_record(
			symbol, parsed, sentences, rejection_reason or "neural_abstention",
			classifier, relation_audit,
		)
	candidate = winner.candidate
	result = candidate.result
	audit = _base_audit(parsed, sentences, classifier)
	audit.update({
		"selection_method": "mathbert_cross_encoder",
		"selected_relation": winner.relation,
		"candidate_phrase": winner.phrase,
		"full_evidence_sentence": candidate.sentence,
		"relation_probabilities": winner.probabilities,
		"cross_encoder_score": winner.cross_encoder_score,
		"component_relations": relation_audit,
		"extractive": winner.phrase in candidate.sentence,
		"rejection_reason": None,
	})
	return SymbolMeaningRecord(
		canonical=symbol["canonical"],
		latex_forms=list(symbol.get("latex_forms", [])),
		aliases=parsed.aliases,
		definition=winner.phrase,
		confidence=round(winner.cross_encoder_score, 4),
		strategy="mathbert_cross_encoder",
		source_text=candidate.sentence,
		source_chunk_id=result.get("chunk_id"),
		source_paragraph_ids=list(result.get("paragraph_ids", [])),
		source_sentence_ids=list(result.get("sentence_ids", [])),
		retrieval_method=result.get("method"),
		retrieval_rank=result.get("rank"),
		retrieval_score=result.get("score"),
		candidate_score=round(winner.cross_encoder_score, 4),
		audit=audit,
	)
