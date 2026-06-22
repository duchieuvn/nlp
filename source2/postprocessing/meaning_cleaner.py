from dataclasses import dataclass, field
import re
from typing import Any


MAX_MEANING_WORDS = 12

LEADING_DROP = re.compile(
	r"^(?:(?:a|an|the|our|their|this|these|those|its)\s+)+",
	re.IGNORECASE,
)
DISCOURSE_PREFIX = re.compile(
	r"^(?:therefore|thus|hence|consequently|furthermore|finally|next|now),?\s+",
	re.IGNORECASE,
)
CLAUSE_BOUNDARY = re.compile(
	r"(?:,\s+(?:we|one|it|they)\b|\s+\b(?:where|which|whose|that|using|"
	r"given\s+by|known\s+as|with\s+\S+\s+being)\b)",
	re.IGNORECASE,
)
TRAILING_REFERENCE = re.compile(
	r"(?:\s*\[\s*[^]]*\s*\]|\s*\(?\s*(?:Eq(?:uation)?|Fig(?:ure)?|"
	r"Sec(?:tion)?|Appendix)\.?\s*\(?\s*\d+[^.;:]*\)?\s*)+$",
	re.IGNORECASE,
)
WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
EQUATION_REFERENCE = re.compile(
	r"(?:\bEq(?:uation)?s?\.?\s*\(?\s*\d+|\\eqref\b)", re.IGNORECASE
)
REFERENCE_BOUNDARY = re.compile(
	r"\s+(?:in|from|using|of)\s+(?:Eq(?:uation)?s?\.?\s*\(?\s*\d+|\\eqref\b)",
	re.IGNORECASE,
)
LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
LEADING_NOISE = re.compile(
	r"^(?:(?:for\s+example|for\s+instance|for)\s+)+",
	re.IGNORECASE,
)
INCOMPLETE_EDGE = re.compile(
	r"^(?:we|one|it|such|to|of|as|and|by|in|from|then|account|taking|resulting|"
	r"operatorname|mathrm|text)\b|"
	r"\b(?:a|an|the|and|as|by|eq|equation|for|from|in|of|or|to|via|with)\s*$",
	re.IGNORECASE,
)

SYMBOL_DEFINITION_START = re.compile(
	r"^\s*(?:where\b|with\s+(?:\\|[A-Za-z](?:\s|[_({]))|"
	r"(?:Here,?|In\s+Eq(?:uation)?\.?[^,]*,)\s+(?:\\|[A-Za-z](?:[_({]|\s+"
	r"(?:is|are|denotes?|represents?))))",
	re.IGNORECASE,
)

SUBJECT_PREDICATE = re.compile(
	r"(?P<phrase>(?:(?:the|a|an|our|this|these|their)\s+)?"
	r"[A-Za-z][^,;:\n]{1,180}?)\s+"
	r"(?:(?:can|may|could|shall|will)\s+be|(?:is|are|was|were))\s+"
	r"(?:readily\s+)?(?:written|given|expressed|defined|represented|described|"
	r"evaluated|obtained|constructed|approximated|interpreted|characterized|"
	r"formulated|calculated)\b",
	re.IGNORECASE,
)
TAKES_FORM = re.compile(
	r"(?P<phrase>(?:(?:the|a|an|our|this|these|their)\s+)?"
	r"[A-Za-z][^,;:\n]{1,160}?)\s+(?:takes?|has)\s+the\s+form\b",
	re.IGNORECASE,
)
REPORTED_SUBJECT = re.compile(
	r"\b(?:one|we)\s+(?:can\s+|may\s+)?(?:observes?|see|find|note)\s+that\s+"
	r"(?P<phrase>(?:(?:the|a|an)\s+)?[^,;:.]{2,140}?)\s+"
	r"(?:commutes?|evolves?|satisfies|obeys|describes?|represents?|is|are|has|holds)\b",
	re.IGNORECASE,
)
ACTIVE_OBJECT = re.compile(
	r"\b(?:we|one)\s+(?:can\s+|may\s+|then\s+)?"
	r"(?:derive|obtain|define|construct|introduce|write|express|represent|formulate|"
	r"yield|give|consider)\s+(?P<phrase>(?:(?:the|a|an)\s+)?[^,;:.]{2,150})",
	re.IGNORECASE,
)
DERIVED_OBJECT = re.compile(
	r"\b(?:derive|obtain|yield|give)s?\s+(?P<phrase>(?:(?:the|a|an)\s+)?"
	r"[^,;:.]{2,140})",
	re.IGNORECASE,
)
EXPLICIT_DESCRIPTION = re.compile(
	r"\bEq(?:uation)?\.?\s*\(?\s*[^)\s]+\s*\)?\s+"
	r"(?:defines|describes|represents|gives|corresponds\s+to)\s+"
	r"(?P<phrase>[^,;:.]{2,140})",
	re.IGNORECASE,
)
NAMED_COMPLEMENT = re.compile(
	r"\b(?:called|known\s+as|serves?\s+as|corresponds?\s+to)\s+"
	r"(?P<phrase>(?:(?:the|a|an)\s+)?[^,;:.]{2,120})",
	re.IGNORECASE,
)

SCIENCE_HEADS = {
	"amplitude", "basis", "channel", "coefficient", "condition", "constraint",
	"correlation", "covariance", "current", "density", "distribution", "dynamics",
	"energy", "entropy", "equation", "error", "evolution", "factor", "fidelity",
	"field", "force", "form", "formula", "function", "gain", "hamiltonian",
	"identity", "impedance", "index", "inequality", "interaction", "kernel",
	"lagrangian", "law", "matrix", "model", "motion", "operator", "potential",
	"probability", "propagator", "relation", "solution", "spectrum", "state",
	"time", "transformation", "vector", "wavefunction", "qubit",
}

FORBIDDEN_VERBS = {
	"am", "are", "be", "been", "being", "can", "called", "commute", "commutes",
	"apply", "applies", "calculate", "calculates", "construct", "constructs",
	"applying", "calculating", "constructing", "counting",
	"carrying", "introducing",
	"consist", "consists", "contain", "contains", "correspond", "corresponds",
	"could", "define", "defines", "denote", "denotes", "depend", "depends", "derive",
	"derives", "defining", "denoting", "deriving", "expand", "expands", "expanding",
	"describe", "describes", "evaluate", "evaluates", "evolve", "evolves", "give",
	"gives", "has", "have", "holds", "indicate", "indicates", "is", "lead",
	"leads", "may", "model", "models", "obey", "obeys", "obtain", "obtains",
	"perform", "performs", "quantize", "quantizes", "refer", "refers", "simplify",
	"simplifies", "using", "performing", "quantizing", "obtaining", "use", "uses",
	"write", "writes", "writing", "yielding", "taking", "resulting", "measuring",
	"observe", "observes", "note", "notes", "find", "finds", "see", "sees",
	"represent", "represents", "satisfies", "serve", "serves", "shall", "should",
	"was", "were", "will", "would", "yield", "yields",
}


@dataclass(frozen=True)
class MeaningCleanup:
	meaning: str
	applied: bool
	strategy: str
	removed_prefix: str = ""
	removed_suffix: str = ""
	marker: str = ""
	candidates: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _word_count(text: str) -> int:
	return len(WORD.findall(text))


def _trim_phrase(raw: str) -> str:
	phrase = re.sub(r"\s+", " ", raw).strip()
	phrase = DISCOURSE_PREFIX.sub("", phrase)
	phrase = LEADING_NOISE.sub("", phrase)
	boundary = CLAUSE_BOUNDARY.search(phrase)
	if boundary:
		phrase = phrase[:boundary.start()]
	reference = REFERENCE_BOUNDARY.search(phrase)
	if reference:
		phrase = phrase[:reference.start()]
	phrase = TRAILING_REFERENCE.sub("", phrase)
	phrase = phrase.strip(" \t\r\n,;:.()")
	phrase = LEADING_DROP.sub("", phrase).strip(" \t\r\n,;:.")
	return phrase


def _validation_reasons(phrase: str, require_science_head: bool = False) -> list[str]:
	reasons = []
	words = [word.casefold() for word in WORD.findall(phrase)]
	if len(words) < 2 and not any(word in SCIENCE_HEADS for word in words):
		reasons.append("too_short")
	if len(words) > MAX_MEANING_WORDS:
		reasons.append("too_long")
	if any(word in FORBIDDEN_VERBS for word in words):
		reasons.append("contains_finite_or_context_verb")
	if CLAUSE_BOUNDARY.search(phrase):
		reasons.append("contains_definition_or_relative_clause")
	if EQUATION_REFERENCE.search(phrase):
		reasons.append("contains_equation_reference")
	if INCOMPLETE_EDGE.search(phrase):
		reasons.append("incomplete_or_invalid_edge")
	for opening, closing in (("(", ")"), ("{", "}"), ("[", "]")):
		if phrase.count(opening) != phrase.count(closing):
			reasons.append("unbalanced_delimiters")
			break
	if require_science_head and not any(word in SCIENCE_HEADS for word in words):
		reasons.append("missing_science_head")
	compact = re.sub(r"\s+", "", phrase)
	alpha_count = len(re.findall(r"[A-Za-z]", compact))
	math_count = len(compact) - alpha_count
	if LATEX_COMMAND.search(phrase) and (len(words) < 2 or math_count > alpha_count * 2):
		reasons.append("math_dominated")
	return reasons


def _candidate_record(
	raw: str,
	strategy: str,
	base_score: int,
	require_science_head: bool = False,
) -> dict[str, Any]:
	phrase = _trim_phrase(raw)
	reasons = _validation_reasons(phrase, require_science_head=require_science_head)
	words = [word.casefold() for word in WORD.findall(phrase)]
	score = base_score
	if any(word in SCIENCE_HEADS for word in words):
		score += 10
	if 2 <= len(words) <= 8:
		score += 3
	if LATEX_COMMAND.search(phrase):
		score -= 1
	return {
		"raw": raw.strip(),
		"phrase": phrase,
		"strategy": strategy,
		"score": score,
		"accepted": not reasons,
		"reasons": reasons,
	}


def _template_candidates(text: str) -> list[dict[str, Any]]:
	candidates = []
	for strategy, pattern, score in (
		("subject_before_introduction", SUBJECT_PREDICATE, 90),
		("subject_takes_form", TAKES_FORM, 88),
		("reported_claim_subject", REPORTED_SUBJECT, 86),
		("active_context_object", ACTIVE_OBJECT, 82),
		("derived_object", DERIVED_OBJECT, 80),
		("explicit_description", EXPLICIT_DESCRIPTION, 84),
		("named_complement", NAMED_COMPLEMENT, 83),
	):
		for match in pattern.finditer(text):
			candidates.append(_candidate_record(match.group("phrase"), strategy, score))
	return candidates


def _existing_phrase_candidate(text: str) -> list[dict[str, Any]]:
	phrase = _candidate_record(text, "existing_phrase", 95)
	return [phrase] if _word_count(phrase["phrase"]) <= MAX_MEANING_WORDS else []


def _fallback_candidates(text: str) -> list[dict[str, Any]]:
	candidates = []
	segments = re.split(
		r"[,;:]|\b(?:where|which|whose|that|because|since|if|then)\b",
		text,
		flags=re.IGNORECASE,
	)
	for segment in segments:
		segment = segment.strip()
		words = list(WORD.finditer(segment))
		if 1 <= len(words) <= MAX_MEANING_WORDS:
			candidates.append(_candidate_record(
				segment, "science_head_segment", 35, require_science_head=True
			))
		for index, word_match in enumerate(words):
			if word_match.group(0).casefold() not in SCIENCE_HEADS:
				continue
			for width in range(1, min(5, index + 1) + 1):
				start_index = index - width + 1
				raw = segment[words[start_index].start():word_match.end()]
				candidates.append(_candidate_record(
					raw,
					"science_head_window",
					40 + width,
					require_science_head=True,
				))
	return candidates


def clean_meaning(text: str, source_strategy: str | None = None) -> MeaningCleanup:
	"""Select a short extractive equation-concept phrase from source text."""
	cleaned = re.sub(r"\s+", " ", text).strip()
	if not cleaned:
		return MeaningCleanup("", False, "empty_source")
	if SYMBOL_DEFINITION_START.search(cleaned):
		candidate = {
			"raw": cleaned,
			"phrase": "",
			"strategy": "symbol_definition_exclusion",
			"score": 0,
			"accepted": False,
			"reasons": ["symbol_definition_sentence"],
		}
		return MeaningCleanup(
			"", True, "no_reliable_phrase", removed_suffix=cleaned,
			candidates=(candidate,),
		)

	candidates = []
	if source_strategy in {
		"anchor_subject", "anchor_object", "derived_object", "explicit_description"
	}:
		candidates.extend(_existing_phrase_candidate(cleaned))
	candidates.extend(_template_candidates(cleaned))
	candidates.extend(_fallback_candidates(cleaned))
	accepted = [candidate for candidate in candidates if candidate["accepted"]]
	if not accepted:
		return MeaningCleanup(
			"", True, "no_reliable_phrase", removed_suffix=cleaned,
			candidates=tuple(candidates),
		)
	selected = max(
		accepted,
		key=lambda candidate: (
			candidate["score"],
			-_word_count(candidate["phrase"]),
			candidate["phrase"],
		),
	)
	meaning = selected["phrase"]
	start = cleaned.find(meaning)
	return MeaningCleanup(
		meaning=meaning,
		applied=meaning != cleaned,
		strategy=selected["strategy"],
		removed_prefix=cleaned[:start].strip() if start >= 0 else "",
		removed_suffix=cleaned[start + len(meaning):].strip() if start >= 0 else "",
		candidates=tuple(candidates),
	)


def postprocess_record(record: dict[str, Any]) -> dict[str, Any]:
	updated = dict(record)
	audit = dict(record.get("audit", {}))
	previous = audit.get("postprocessing", {})
	original = str(
		previous.get("original_meaning")
		if previous.get("applied") and previous.get("original_meaning")
		else record.get("meaning", "")
	)
	result = clean_meaning(original, source_strategy=record.get("strategy"))
	flagged = bool(original) and not result.meaning
	output_meaning = original if flagged else result.meaning
	applied = result.applied and not flagged
	updated["meaning"] = output_meaning
	audit["postprocessing"] = {
		"applied": applied,
		"flagged": flagged,
		"strategy": result.strategy,
		"original_meaning": original if result.applied or flagged else None,
		"removed_prefix": result.removed_prefix or None,
		"removed_suffix": result.removed_suffix or None,
		"marker": result.marker or None,
		"candidates": list(result.candidates),
		"extractive": bool(output_meaning) and output_meaning in str(
			record.get("source_text", "")
		),
	}
	updated["audit"] = audit
	return updated
