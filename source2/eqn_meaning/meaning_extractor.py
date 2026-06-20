from candidates import collect_candidates
from meaning_config import MINIMUM_CANDIDATE_SCORE
from meaning_models import MeaningRecord
from meaning_scorer import score_candidate
from patterns import (
	DEFINITION_CUE,
	INCOMPLETE_ENDING,
	NAMED_CONCEPT,
	PROCEDURAL_FRAGMENT,
	extract_span,
)


def _empty_record(equation_id: str, equation: str, candidate_count: int) -> MeaningRecord:
	return MeaningRecord(
		equation_id=equation_id,
		equation=equation,
		meaning="",
		confidence=0.0,
		strategy="no_reliable_candidate",
		source_text="",
		source_chunk_id=None,
		audit={
			"candidate_count": candidate_count,
			"minimum_candidate_score": MINIMUM_CANDIDATE_SCORE,
		},
	)


def extract_equation_meaning(query: dict, equation: str, reranker=None) -> MeaningRecord:
	equation_id = query["equation_id"]
	candidates = collect_candidates(query)
	maximum_retrieval_score = max(
		(result["score"] for result in query["results"]),
		default=0.0,
	)
	eligible = []
	for candidate in candidates:
		meaning, strategy = extract_span(
			candidate.text,
			equation_id,
			candidate.position,
		)
		# word_count = len(candidate.text.split())
		# if word_count < 4 or word_count > 60:
		# 	continue
		hard_filter_reasons = []
		if strategy == "source_sentence" and (
			PROCEDURAL_FRAGMENT.search(candidate.text)
			or INCOMPLETE_ENDING.search(candidate.text)
		):
			hard_filter_reasons.append("procedural_or_incomplete")
		if not (
			DEFINITION_CUE.search(candidate.text)
			or NAMED_CONCEPT.search(candidate.text)
			or candidate.position == "before_equation"
		):
			hard_filter_reasons.append("missing_cue_or_before_equation_anchor")
		eligible.append((candidate, meaning, strategy, hard_filter_reasons))
	mathbert_scores = (
		reranker.score_candidates(
			equation,
			query,
			[candidate for candidate, _, _, _ in eligible],
		)
		if reranker is not None else [None] * len(eligible)
	)
	if len(mathbert_scores) != len(eligible):
		raise ValueError("MathBERT reranker returned an invalid number of scores")
	if reranker is not None and mathbert_scores:
		minimum_mathbert = min(mathbert_scores)
		maximum_mathbert = max(mathbert_scores)
		mathbert_range = maximum_mathbert - minimum_mathbert
		normalized_mathbert_scores = [
			(score - minimum_mathbert) / mathbert_range
			if mathbert_range > 1e-8 else 0.5
			for score in mathbert_scores
		]
	else:
		normalized_mathbert_scores = [None] * len(eligible)

	scored = []
	for (
		candidate, meaning, strategy, hard_filter_reasons
	), mathbert_similarity, mathbert_score in zip(
		eligible, mathbert_scores, normalized_mathbert_scores
	):
		score, features = score_candidate(
			candidate,
			equation_id,
			maximum_retrieval_score,
			mathbert_score,
		)
		scored.append((
			score, candidate, features, meaning, strategy,
			mathbert_similarity, mathbert_score, hard_filter_reasons,
		))
	if not scored:
		return _empty_record(equation_id, equation, len(candidates))

	(
		score, candidate, features, meaning, strategy,
		mathbert_similarity, mathbert_score, hard_filter_reasons,
	) = max(
		scored,
		key=lambda item: (
			item[0],
			-item[1].result["rank"],
			item[1].text,
		),
	)
	selection_method = "combined_score"
	minimum_score_bypassed = False
	if score < MINIMUM_CANDIDATE_SCORE:
		if reranker is None:
			return _empty_record(equation_id, equation, len(candidates))
		(
			score, candidate, features, meaning, strategy,
			mathbert_similarity, mathbert_score, hard_filter_reasons,
		) = max(
			scored,
			key=lambda item: (
				item[5],
				-item[1].result["rank"],
				item[1].text,
			),
		)
		selection_method = "mathbert_fallback"
		minimum_score_bypassed = True

	result = candidate.result
	return MeaningRecord(
		equation_id=equation_id,
		equation=equation,
		meaning=meaning,
		confidence=round(min(1.0, score / 12.0), 4),
		strategy=strategy,
		source_text=candidate.text,
		source_chunk_id=result["chunk_id"],
		source_paragraph_ids=result["paragraph_ids"],
		source_sentence_ids=result["sentence_ids"],
		retrieval_method=result["method"],
		retrieval_rank=result["rank"],
		retrieval_score=result["score"],
		candidate_score=round(score, 4),
		audit={
			"position": candidate.position,
			"score_features": features,
			"candidate_count": len(candidates),
			"minimum_candidate_score": MINIMUM_CANDIDATE_SCORE,
			"selection_method": selection_method,
			"minimum_score_bypassed": minimum_score_bypassed,
			"extractive": meaning in candidate.text,
			"mathbert_model": getattr(reranker, "model_name", None),
			"mathbert_similarity": (
				round(mathbert_similarity, 6)
				if mathbert_similarity is not None else None
			),
			"mathbert_normalized_score": (
				round(mathbert_score, 6) if mathbert_score is not None else None
			),
			"hard_filtered": bool(hard_filter_reasons),
			"hard_filter_reasons": hard_filter_reasons,
			"hard_filtered_paper_id": (
				result["paper_id"] if hard_filter_reasons else None
			),
		},
	)
