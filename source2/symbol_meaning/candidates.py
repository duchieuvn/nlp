from dataclasses import dataclass
import re


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class SymbolDefinitionCandidate:
	text: str
	result: dict


def collect_candidates(results: list[dict]) -> list[SymbolDefinitionCandidate]:
	by_text: dict[str, SymbolDefinitionCandidate] = {}
	for result in results:
		parts = (
			[result["text"].strip()]
			if result["chunk_type"] == "sentence"
			else SENTENCE_BOUNDARY.split(result["text"].strip())
		)
		for text in parts:
			text = text.strip()
			if not text or text.startswith("Section:") or text.startswith("Equation ("):
				continue
			candidate = SymbolDefinitionCandidate(text, result)
			key = re.sub(r"\s+", " ", text).casefold()
			current = by_text.get(key)
			if current is None or _priority(candidate) > _priority(current):
				by_text[key] = candidate
	return list(by_text.values())


def _priority(candidate: SymbolDefinitionCandidate) -> tuple[int, float]:
	return (
		int(candidate.result["chunk_type"] == "sentence"),
		candidate.result["score"],
	)
