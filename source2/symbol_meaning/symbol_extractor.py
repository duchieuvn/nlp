from candidates import collect_candidates
from patterns import extract_definition, mentions_alias
from symbol_config import MINIMUM_DEFINITION_SCORE
from symbol_models import SymbolMeaningRecord
from symbol_scorer import score_candidate
from spacy_fallback import extract_dependency_definition


def _empty_record(symbol: dict, candidate_count: int) -> SymbolMeaningRecord:
	return SymbolMeaningRecord(
		canonical=symbol["canonical"],
		latex_forms=list(symbol.get("latex_forms", [])),
		aliases=list(symbol.get("aliases", [])),
		definition="",
		confidence=0.0,
		strategy="no_reliable_definition",
		source_text="",
		source_chunk_id=None,
		audit={
			"candidate_count": candidate_count,
			"minimum_definition_score": MINIMUM_DEFINITION_SCORE,
		},
	)


def extract_symbol_meaning(
	symbol: dict,
	equation_id: str,
	results: list[dict],
	nlp=None,
) -> SymbolMeaningRecord:
	aliases = list(dict.fromkeys(symbol.get("aliases", [])))
	candidates = collect_candidates(results)
	maximum_retrieval_score = max(
		(result["score"] for result in results),
		default=0.0,
	)
	scored = []
	for candidate in candidates:
		if not mentions_alias(candidate.text, aliases):
			continue
		definition, strategy, matched_alias = extract_definition(
			candidate.text, aliases
		)
		if not definition:
			definition, matched_alias = extract_dependency_definition(
				candidate.text, aliases, nlp
			)
			if definition:
				strategy = "spacy_dependency"
		if not definition:
			continue
		score, features = score_candidate(
			candidate,
			equation_id,
			maximum_retrieval_score,
			strategy,
		)
		scored.append((
			score, candidate, definition, strategy, matched_alias, features
		))
	if not scored:
		return _empty_record(symbol, len(candidates))

	score, candidate, definition, strategy, matched_alias, features = max(
		scored,
		key=lambda item: (
			item[0],
			-item[1].result["rank"],
			item[2],
		),
	)
	if score < MINIMUM_DEFINITION_SCORE:
		return _empty_record(symbol, len(candidates))

	result = candidate.result
	return SymbolMeaningRecord(
		canonical=symbol["canonical"],
		latex_forms=list(symbol.get("latex_forms", [])),
		aliases=aliases,
		definition=definition,
		confidence=round(min(1.0, score / 11.0), 4),
		strategy=strategy,
		source_text=candidate.text,
		source_chunk_id=result["chunk_id"],
		source_paragraph_ids=list(result.get("paragraph_ids", [])),
		source_sentence_ids=list(result.get("sentence_ids", [])),
		retrieval_method=result["method"],
		retrieval_rank=result["rank"],
		retrieval_score=result["score"],
		candidate_score=round(score, 4),
		audit={
			"matched_alias": matched_alias,
			"score_features": features,
			"candidate_count": len(candidates),
			"minimum_definition_score": MINIMUM_DEFINITION_SCORE,
			"extractive": definition in candidate.text,
		},
	)
