import re

from candidates import MeaningCandidate
from meaning_config import MATHBERT_SCORE_WEIGHT
from patterns import DEFINITION_CUE, NAMED_CONCEPT


def _mentions_equation(text: str, equation_id: str) -> bool:
	return bool(re.search(
		rf"\b(?:Eq(?:uation)?s?\.?|Equation)\s*\(?\s*"
		rf"{re.escape(equation_id)}(?:\s*[,)-]|\s*$)",
		text,
		re.IGNORECASE,
	))


def score_candidate(
	candidate: MeaningCandidate,
	equation_id: str,
	maximum_retrieval_score: float,
	mathbert_score: float | None = None,
) -> tuple[float, dict[str, float]]:
	text = candidate.text
	result = candidate.result
	features: dict[str, float] = {}
	if _mentions_equation(text, equation_id):
		features["explicit_equation_reference"] = 5.0
	if DEFINITION_CUE.search(text):
		features["definition_cue"] = 3.0
	if NAMED_CONCEPT.search(text):
		features["named_concept"] = 3.0
	if candidate.position in {"before_equation", "after_equation"}:
		features["equation_adjacent"] = 2.0
	if result["chunk_id"].endswith(f":equation_neighborhood:{equation_id}"):
		features["exact_equation_neighborhood"] = 2.0
	if maximum_retrieval_score > 0:
		features["retrieval_score"] = result["score"] / maximum_retrieval_score
	if mathbert_score is not None:
		features["mathbert_similarity"] = mathbert_score * MATHBERT_SCORE_WEIGHT
	word_count = len(text.split())
	if word_count < 4 or word_count > 60:
		features["length_penalty"] = -1.0
	return sum(features.values()), features
