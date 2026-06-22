import re

from .patterns import alias_pattern, clean_definition, is_valid_definition
from .symbol_models import PhraseCandidate


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def collect_sentences(results: list[dict]) -> list[tuple[str, dict]]:
	by_text = {}
	for result in results:
		parts = (
			[result["text"].strip()]
			if result.get("chunk_type") == "sentence"
			else SENTENCE_BOUNDARY.split(result["text"].strip())
		)
		for text in parts:
			text = text.strip()
			if not text or text.startswith(("Section:", "Equation (")):
				continue
			key = re.sub(r"\s+", " ", text).casefold()
			current = by_text.get(key)
			priority = (result.get("chunk_type") == "sentence", result.get("score", 0))
			if current is None or priority > current[2]:
				by_text[key] = (text, result, priority)
	return [(text, result) for text, result, _ in by_text.values()]


def _regex_phrases(text: str, aliases: list[str]) -> list[tuple[str, str]]:
	alias = alias_pattern(aliases)
	if alias == "(?:)":
		return []
	patterns = (
		("cue_tail", rf"{alias}\s+(?:denotes?|represents?|means?|refers?\s+to|stands?\s+for|is\s+defined\s+as|is|are)\s+(?P<phrase>[^.;]{{1,220}})"),
		("let_be", rf"\blet\s+{alias}\s+be\s+(?P<phrase>[^.;]{{1,220}})"),
		("reverse_definition", rf"(?:denote|define|represent|write|call)\s+(?P<phrase>[^.;]{{1,220}}?)\s+(?:as|by)\s+{alias}"),
		("coordinated_right", rf"\band\s+(?P<phrase>[^.;,]{{1,100}}?)\s+as\s+{alias}"),
		("noun_before_symbol", rf"(?P<phrase>(?:the|a|an)\s+[A-Za-z][A-Za-z0-9' -]{{2,100}}?)\s+{alias}(?!\s*(?:is|are|=))"),
		("appositive", rf"{alias}\s*,\s*(?P<phrase>(?:the|a|an)\s+[^,.;]{{1,100}})"),
	)
	output = []
	for source, pattern in patterns:
		for match in re.finditer(pattern, text, re.IGNORECASE):
			phrase = clean_definition(match.group("phrase"))
			if is_valid_definition(phrase, aliases):
				output.append((phrase, source))
	return output


def extract_phrase_candidates(
	sentences: list[tuple[str, dict]],
	aliases: list[str],
	nlp=None,
) -> list[PhraseCandidate]:
	output = []
	seen = set()
	for text, result in sentences:
		phrases = _regex_phrases(text, aliases)
		if nlp is not None:
			try:
				phrases.extend((chunk.text, "noun_chunk") for chunk in nlp(text).noun_chunks)
			except (AttributeError, ValueError):
				pass
		for raw_phrase, source in phrases:
			phrase = clean_definition(raw_phrase)
			key = (phrase.casefold(), re.sub(r"\s+", " ", text).casefold())
			if key in seen or not is_valid_definition(phrase, aliases):
				continue
			seen.add(key)
			output.append(PhraseCandidate(phrase, text, result, source))
	return output
