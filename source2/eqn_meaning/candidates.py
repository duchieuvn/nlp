from dataclasses import dataclass
import re


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class MeaningCandidate:
	text: str
	position: str
	result: dict


def _split_sentences(text: str) -> list[str]:
	return [
		part.strip()
		for part in SENTENCE_BOUNDARY.split(text.strip())
		if part.strip()
	]


def candidates_from_result(result: dict, equation_id: str) -> list[MeaningCandidate]:
	lines = [line.strip() for line in result["text"].splitlines() if line.strip()]
	equation_line = re.compile(
		rf"^Equation\s*\(\s*{re.escape(equation_id)}\s*\)\s*:",
		re.IGNORECASE,
	)
	target_line_index = next(
		(index for index, line in enumerate(lines) if equation_line.match(line)),
		None,
	)
	candidates = []
	for line_index, line in enumerate(lines):
		if line.startswith("Section:") or line.startswith("Equation ("):
			continue
		sentences = _split_sentences(line)
		for sentence_index, sentence in enumerate(sentences):
			position = "retrieved"
			if target_line_index is not None:
				if line_index == target_line_index - 1 and sentence_index == len(sentences) - 1:
					position = "before_equation"
				elif line_index == target_line_index + 1 and sentence_index == 0:
					position = "after_equation"
			candidates.append(MeaningCandidate(sentence, position, result))
	return candidates


def collect_candidates(query: dict) -> list[MeaningCandidate]:
	by_text: dict[str, MeaningCandidate] = {}
	for result in query["results"]:
		for candidate in candidates_from_result(result, query["equation_id"]):
			key = re.sub(r"\s+", " ", candidate.text).casefold()
			current = by_text.get(key)
			if current is None or _candidate_priority(candidate) > _candidate_priority(current):
				by_text[key] = candidate
	return list(by_text.values())


def _candidate_priority(candidate: MeaningCandidate) -> tuple[int, float]:
	is_adjacent = candidate.position in {"before_equation", "after_equation"}
	is_exact_neighborhood = candidate.result["chunk_id"].endswith(
		f":equation_neighborhood:{candidate.result['nearby_equation_ids'][0]}"
	) if candidate.result["nearby_equation_ids"] else False
	return (
		int(is_adjacent) + int(is_exact_neighborhood),
		candidate.result["score"],
	)
