from candidates import SymbolDefinitionCandidate


def score_candidate(
	candidate: SymbolDefinitionCandidate,
	equation_id: str,
	maximum_retrieval_score: float,
	strategy: str,
) -> tuple[float, dict[str, float]]:
	result = candidate.result
	features: dict[str, float] = (
		{"spacy_dependency_pattern": 4.0, "copular_or_appositive_cue": 2.0}
		if strategy == "spacy_dependency"
		else {"alias_definition_pattern": 4.0, "definition_cue": 2.0}
	)
	if equation_id in result.get("nearby_equation_ids", []):
		features["near_target_equation"] = 2.0
	if result["chunk_type"] == "sentence":
		features["sentence_chunk"] = 1.0
	if strategy == "definition_before_symbol":
		features["explicit_definition_construction"] = 1.0
	if maximum_retrieval_score > 0:
		features["retrieval_score"] = result["score"] / maximum_retrieval_score
	return sum(features.values()), features
