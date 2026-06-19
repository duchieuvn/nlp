"""Extract equation meanings from scientific-text context with SciSpaCy.

The module identifies noun phrases associated with an ``[EQUATION]`` marker,
stores the selected phrase as the equation meaning, and records an audit trail.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import re
import sys
import spacy
from config import EQUATION_MEANINGS_FILE, EQUATIONS_FILE


EQUATION_MARKER = "[EQUATION]"
MODEL_CONTEXT_WORD_LIMITS = (40, 25, 15, 8)
SPACY_MODEL = "en_core_sci_scibert"
EXPAND_PREPOSITIONS = {"of", "for", "with", "in", "on"}
EXPAND_STOP_POS = {"AUX", "CCONJ", "PUNCT", "SCONJ", "VERB"}
SCIBERT_INSTALL_URL = (
	"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
	"en_core_sci_scibert-0.5.4.tar.gz"
)
INSTALL_MESSAGE = (
	"SciSpaCy/SciBERT is required for step4_eqn_meaning.py. "
	f"Install it with: pip install {SCIBERT_INSTALL_URL}"
)


@dataclass(frozen=True)
class CandidateSpan:
	"""A scored noun-phrase candidate.

	Attributes
	----------
	score : tuple[int, int, int, int, int]
		Ranking components used to compare candidates.
	start : int
		Start token index in the parsed document.
	end : int
		Exclusive end token index in the parsed document.
	text : str
		Cleaned candidate text.
	"""

	score: tuple[int, int, int, int, int]
	start: int
	end: int
	text: str


ANCHOR_PATTERNS = (
	("can be written as", re.compile(r"\bcan\s+be\s+written\s+as\b", re.I)),
	("is written as", re.compile(r"\bis\s+written\s+as\b", re.I)),
	("is given by", re.compile(r"\bis\s+given\s+by\b", re.I)),
	("takes the form", re.compile(r"\btakes\s+the\s+form\b", re.I)),
	("defined as", re.compile(r"\bdefined\s+as\b", re.I)),
	("reads", re.compile(r"\breads\b", re.I)),
	("satisfies", re.compile(r"\bsatisfies\b", re.I)),
	("describes", re.compile(r"\bdescribes\b", re.I)),
	("represented by", re.compile(r"\brepresented\s+by\b", re.I)),
	("represented", re.compile(r"\brepresented\b", re.I)),
	("obtained", re.compile(r"\bobtained\s+by\b", re.I)),
)

SCIENCE_HEAD_WORDS = {
	"condition",
	"constraint",
	"equation",
	"expression",
	"formula",
	"function",
	"hamiltonian",
	"identity",
	"inequality",
	"lagrangian",
	"law",
	"matrix",
	"model",
	"operator",
	"relation",
	"state",
	"theorem",
}

GENERIC_CANDIDATES = {
	"condition",
	"constraint",
	"equation",
	"expression",
	"formula",
	"function",
	"law",
	"matrix",
	"model",
	"operator",
	"relation",
	"state",
}

CONTEXT_VERB_LEMMAS = {
	"compute",
	"denote",
	"evaluate",
	"lead",
	"obtain",
	"represent",
	"yield",
}

CONTEXT_VERB_FORMS = CONTEXT_VERB_LEMMAS | {
	"computed",
	"computes",
	"computing",
	"denoted",
	"denotes",
	"denoting",
	"evaluated",
	"evaluates",
	"evaluating",
	"leads",
	"leading",
	"obtained",
	"obtains",
	"obtaining",
	"represented",
	"represents",
	"representing",
	"yielded",
	"yields",
	"yielding",
}

LEADING_DROP_WORDS = {
	"a",
	"an",
	"our",
	"the",
	"their",
	"these",
	"this",
	"those",
}


def load_spacy_model():
	"""Load SciSpaCy's SciBERT pipeline and register the equation marker.

	Returns
	-------
	spacy.Language
		Configured SciSpaCy language pipeline.

	Raises
	------
	RuntimeError
		If spaCy or the required SciBERT model is unavailable.
	"""
	try:
		import spacy
		from spacy.symbols import ORTH
	except ModuleNotFoundError as exc:
		raise RuntimeError(INSTALL_MESSAGE) from exc

	try:
		nlp = spacy.load(SPACY_MODEL)
	except OSError as exc:
		raise RuntimeError(INSTALL_MESSAGE) from exc

	nlp.tokenizer.add_special_case(EQUATION_MARKER, [{ORTH: EQUATION_MARKER}])
	return nlp


def normalize_text(text: str) -> str:
	"""Collapse whitespace and trim a text string.

	Parameters
	----------
	text : str
		Text to normalize.

	Returns
	-------
	str
		Normalized text.
	"""
	return re.sub(r"\s+", " ", text).strip()


def clean_model_text(text: str) -> str:
	"""Remove LaTeX syntax that is unnecessary for contextual parsing.

	Parameters
	----------
	text : str
		Raw equation context.

	Returns
	-------
	str
		Whitespace-normalized plain text.
	"""
	text = re.sub(r"\\[A-Za-z]+", " ", text)
	text = re.sub(r"[{}_^]", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def model_input_window(window: str, words_per_side: int) -> str:
	"""Create a bounded model input around the equation marker.

	Parameters
	----------
	window : str
		Full surrounding-text window.
	words_per_side : int
		Maximum words retained on each side of the marker.

	Returns
	-------
	str
		Cleaned and cropped model input.
	"""
	window = normalize_text(window)
	if EQUATION_MARKER not in window:
		return clean_model_text(" ".join(window.split()[: words_per_side * 2]))

	before, after = window.split(EQUATION_MARKER, 1)
	before_words = before.split()[-words_per_side:]
	after_words = after.split()[:words_per_side]
	return clean_model_text(
		" ".join(before_words + [EQUATION_MARKER] + after_words)
	)


def is_transformer_length_error(exc: Exception) -> bool:
	"""Check whether an exception represents SciBERT's token limit error.

	Parameters
	----------
	exc : Exception
		Exception raised while processing text.

	Returns
	-------
	bool
		Whether the message indicates a 512-token tensor mismatch.
	"""
	message = str(exc)
	return "512" in message and "tensor" in message


def parse_window(nlp: spacy.Language, window: str):
	"""Parse a context window, shortening it when SciBERT exceeds its limit.

	Parameters
	----------
	nlp : spacy.Language
		Loaded SciSpaCy pipeline.
	window : str
		Surrounding text containing the equation marker.

	Returns
	-------
	tuple[spacy.tokens.Doc, int]
		Parsed document and the successful words-per-side limit.

	Raises
	------
	RuntimeError
		If all configured context sizes exceed the model limit.
	Exception
		Any processing error unrelated to input length.
	"""
	last_error = None

	for words_per_side in MODEL_CONTEXT_WORD_LIMITS:
		model_text = model_input_window(window, words_per_side)
		try:
			return nlp(model_text), words_per_side
		except Exception as exc:
			if not is_transformer_length_error(exc):
				raise
			last_error = exc

	raise RuntimeError(
		"SciBERT could not process this equation window even after shortening it."
	) from last_error


def clean_candidate(text: str) -> str:
	"""Remove boundary punctuation and leading determiners from a phrase.

	Parameters
	----------
	text : str
		Candidate phrase.

	Returns
	-------
	str
		Cleaned candidate phrase.
	"""
	text = normalize_text(text)
	text = re.sub(r"^[,;:.\s]+|[,;:.\s]+$", "", text)
	words = text.split()
	while words and words[0].lower() in LEADING_DROP_WORDS:
		words.pop(0)
	return " ".join(words)


def candidate_words(text: str) -> list[str]:
	"""Return lowercase alphabetic words from candidate text.

	Parameters
	----------
	text : str
		Candidate text.

	Returns
	-------
	list[str]
		Extracted lowercase words.
	"""
	return [word.lower() for word in re.findall(r"[A-Za-z]+", text)]


def marker_index(doc: spacy.tokens.Doc) -> int | None:
	"""Find the equation marker token in a document.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.

	Returns
	-------
	int or None
		Marker token index, or ``None`` when absent.
	"""
	for token in doc:
		if token.text == EQUATION_MARKER:
			return token.i
	return None


def has_science_head(text: str) -> bool:
	"""Check whether text contains a recognized scientific head word.

	Parameters
	----------
	text : str
		Candidate phrase.

	Returns
	-------
	bool
		Whether a scientific head word is present.
	"""
	words = candidate_words(text)
	return any(word in SCIENCE_HEAD_WORDS for word in words)


def is_generic_candidate(text: str) -> bool:
	"""Check whether text is a single generic scientific noun.

	Parameters
	----------
	text : str
		Candidate phrase.

	Returns
	-------
	bool
		Whether the candidate is generic.
	"""
	words = candidate_words(text)
	return len(words) == 1 and words[0] in GENERIC_CANDIDATES


def is_reference_candidate(text: str) -> bool:
	"""Check whether text resembles a numbered cross-reference.

	Parameters
	----------
	text : str
		Candidate phrase.

	Returns
	-------
	bool
		Whether the phrase resembles an equation or section reference.
	"""
	return bool(re.search(r"\b(?:eq|fig|sec|app|ref)s?\.?\s*\(?\s*\d+", text, re.I))


def is_symbol_heavy_candidate(text: str) -> bool:
	"""Check whether a candidate contains too little natural language.

	Parameters
	----------
	text : str
		Candidate phrase.

	Returns
	-------
	bool
		Whether symbols dominate the candidate.
	"""
	if re.search(r"[()[\]{}]", text):
		return True
	compact = re.sub(r"\s+", "", text)
	if not compact:
		return True
	alpha_count = len(re.findall(r"[A-Za-z]", compact))
	symbol_count = len(compact) - alpha_count
	return alpha_count < 3 or (symbol_count > alpha_count and alpha_count < 8)


def is_reliable_candidate(text: str, allow_generic: bool = False) -> bool:
	"""Validate an extracted equation-name candidate.

	Parameters
	----------
	text : str
		Candidate phrase.
	allow_generic : bool, default=False
		Whether a single generic noun is acceptable.

	Returns
	-------
	bool
		Whether the candidate passes the reliability filters.
	"""
	words = candidate_words(text)
	if not words or len(words) > 12:
		return False
	if is_reference_candidate(text) or is_symbol_heavy_candidate(text):
		return False
	if is_generic_candidate(text) and not allow_generic:
		return False
	return True


def noun_chunk_score(chunk_text: str) -> tuple[int, int]:
	"""Score a noun chunk by scientific relevance and length.

	Parameters
	----------
	chunk_text : str
		Noun-chunk text.

	Returns
	-------
	tuple[int, int]
		Science-head indicator and word count.
	"""
	words = candidate_words(chunk_text)
	has_science_head = any(word in SCIENCE_HEAD_WORDS for word in words)
	return (1 if has_science_head else 0, len(words))


def is_anchor_start(doc: spacy.tokens.Doc, token_index: int, max_end_token_index: int) -> bool:
	"""Check whether an anchor phrase starts at a token index.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	token_index : int
		Potential anchor start index.
	max_end_token_index : int
		Exclusive boundary for matching.

	Returns
	-------
	bool
		Whether a configured anchor starts at the index.
	"""
	text = doc[token_index:max_end_token_index].text
	return any(pattern.match(text) for _, pattern in ANCHOR_PATTERNS)


def expansion_stop_index(doc: spacy.tokens.Doc, start_token_index: int, max_end_token_index: int) -> int:
	"""Find the token boundary for a noun-chunk prepositional expansion.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	start_token_index : int
		First token considered for expansion.
	max_end_token_index : int
		Exclusive upper boundary.

	Returns
	-------
	int
		Exclusive expansion end index.
	"""
	token_index = start_token_index
	while token_index < max_end_token_index:
		token = doc[token_index]
		if token.text == EQUATION_MARKER:
			break
		if is_anchor_start(doc, token_index, max_end_token_index):
			break
		if token.pos_ in EXPAND_STOP_POS:
			break
		token_index += 1
	return token_index


def expand_noun_chunk(chunk: spacy.tokens.Span, doc: spacy.tokens.Doc, max_end_token_index: int) -> tuple[int, int, str]:
	"""Extend a noun chunk through supported prepositional phrases.

	Parameters
	----------
	chunk : spacy.tokens.Span
		Initial noun chunk.
	doc : spacy.tokens.Doc
		Parsed context document.
	max_end_token_index : int
		Exclusive expansion boundary.

	Returns
	-------
	tuple[int, int, str]
		Start index, exclusive end index, and cleaned expanded text.
	"""
	end = chunk.end

	while end < max_end_token_index:
		if doc[end].lower_ not in EXPAND_PREPOSITIONS:
			break

		next_end = expansion_stop_index(doc, end + 1, max_end_token_index)
		if next_end <= end + 1:
			break
		end = next_end

	return chunk.start, end, clean_candidate(doc[chunk.start:end].text)


def expanded_noun_chunk_score(
	candidate: str,
	chunk_end: int,
	expanded_end: int,
	reference_token_index: int,
	same_sentence: bool,
) -> tuple[int, int, int, int, int]:
	"""Build the ranking tuple for an expanded noun chunk.

	Parameters
	----------
	candidate : str
		Cleaned candidate text.
	chunk_end : int
		Original noun-chunk end index.
	expanded_end : int
		Expanded candidate end index.
	reference_token_index : int
		Index used to calculate candidate proximity.
	same_sentence : bool
		Whether the candidate occurs in the reference sentence.

	Returns
	-------
	tuple[int, int, int, int, int]
		Lexicographically comparable ranking tuple.
	"""
	word_count = len(candidate_words(candidate))
	return (
		1 if has_science_head(candidate) else 0,
		1 if same_sentence else 0,
		1 if expanded_end > chunk_end else 0,
		min(word_count, 8),
		-(reference_token_index - expanded_end),
	)


def best_noun_chunk_before(doc: spacy.tokens.Doc, end_token_index: int) -> str:
	"""Select the highest-scoring noun chunk before a boundary.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	end_token_index : int
		Exclusive candidate boundary.

	Returns
	-------
	str
		Best candidate text, or an empty string when none exists.
	"""
	candidates = []

	for chunk in doc.noun_chunks:
		if chunk.end > end_token_index:
			continue
		_, expanded_end, candidate = expand_noun_chunk(chunk, doc, end_token_index)
		if not candidate:
			continue
		score = expanded_noun_chunk_score(
			candidate,
			chunk.end,
			expanded_end,
			end_token_index,
			same_sentence=True,
		)
		candidates.append((score, chunk.start, candidate))

	if not candidates:
		return ""

	candidates.sort(key=lambda item: (item[0], item[1]))
	return candidates[-1][2]


def sentence_bounds_for_marker(doc: spacy.tokens.Doc, marker: int) -> tuple[int, int]:
	"""Return sentence token bounds for the equation marker.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	marker : int
		Equation marker token index.

	Returns
	-------
	tuple[int, int]
		Inclusive start and exclusive end token indices.
	"""
	try:
		sentence = doc[marker].sent
		return sentence.start, sentence.end
	except ValueError:
		return 0, len(doc)


def source_text_for_candidate(doc: spacy.tokens.Doc, start: int, end: int, marker: int) -> str:
	"""Create a compact audit excerpt around a candidate.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	start : int
		Candidate start token index.
	end : int
		Candidate exclusive end token index.
	marker : int
		Equation marker token index.

	Returns
	-------
	str
		Normalized source excerpt ending at the marker.
	"""
	left = doc[max(0, start - 8) : start].text
	candidate = doc[start:end].text
	right = doc[end : min(len(doc), marker)].text
	return normalize_text(" ".join([left, candidate, right, EQUATION_MARKER]))


def scored_noun_chunks_before(doc: spacy.tokens.Doc, marker: int) -> list[CandidateSpan]:
	"""Collect and rank reliable noun chunks before the equation marker.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	marker : int
		Equation marker token index.

	Returns
	-------
	list[CandidateSpan]
		Candidates sorted from highest to lowest score.
	"""
	sentence_start, sentence_end = sentence_bounds_for_marker(doc, marker)
	candidates = []

	for chunk in doc.noun_chunks:
		if chunk.end > marker:
			continue
		expanded_start, expanded_end, candidate = expand_noun_chunk(chunk, doc, marker)
		if not candidate or not is_reliable_candidate(candidate):
			continue

		same_sentence = sentence_start <= chunk.start < sentence_end
		score = expanded_noun_chunk_score(
			candidate,
			chunk.end,
			expanded_end,
			marker,
			same_sentence,
		)
		candidates.append(CandidateSpan(score, expanded_start, expanded_end, candidate))

	return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def sentence_has_context_verb(doc: spacy.tokens.Doc, marker: int) -> bool:
	"""Check for an equation-introducing verb before the marker.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	marker : int
		Equation marker token index.

	Returns
	-------
	bool
		Whether a recognized context verb occurs in the marker sentence.
	"""
	sentence_start, _ = sentence_bounds_for_marker(doc, marker)
	for token in doc[sentence_start:marker]:
		token_text = token.text.lower()
		if token.lemma_.lower() in CONTEXT_VERB_LEMMAS:
			return True
		if token_text in CONTEXT_VERB_FORMS:
			return True
	return False


def pre_marker_noun_chunk_candidate(doc: spacy.tokens.Doc, marker: int) -> tuple[str, dict] | None:
	"""Find a nearby scientific noun chunk before the equation marker.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	marker : int
		Equation marker token index.

	Returns
	-------
	tuple[str, dict] or None
		Candidate text and audit metadata, or ``None`` if no match exists.
	"""
	sentence_start, sentence_end = sentence_bounds_for_marker(doc, marker)

	for candidate in scored_noun_chunks_before(doc, marker):
		distance = marker - candidate.end
		if not (sentence_start <= candidate.start < sentence_end):
			continue
		if distance > 12:
			continue
		if not has_science_head(candidate.text):
			continue

		return candidate.text, {
			"candidate": candidate.text,
			"confidence": "medium",
			"strategy": "pre_marker_noun_chunk",
			"source_text": source_text_for_candidate(
				doc,
				candidate.start,
				candidate.end,
				marker,
			),
		}

	return None


def dependency_context_candidate(doc: spacy.tokens.Doc, marker: int) -> tuple[str, dict] | None:
	"""Find a scientific noun chunk supported by a context verb.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context document.
	marker : int
		Equation marker token index.

	Returns
	-------
	tuple[str, dict] or None
		Candidate text and audit metadata, or ``None`` if no match exists.
	"""
	if not sentence_has_context_verb(doc, marker):
		return None

	sentence_start, sentence_end = sentence_bounds_for_marker(doc, marker)
	for candidate in scored_noun_chunks_before(doc, marker):
		if not (sentence_start <= candidate.start < sentence_end):
			continue
		if marker - candidate.end > 24:
			continue
		if not has_science_head(candidate.text):
			continue

		return candidate.text, {
			"candidate": candidate.text,
			"confidence": "medium",
			"strategy": "dependency_context",
			"source_text": source_text_for_candidate(
				doc,
				candidate.start,
				candidate.end,
				marker,
			),
		}

	return None


def fallback_phrase_before(text: str) -> str:
	"""Extract a science-headed phrase from the end of plain text.

	Parameters
	----------
	text : str
		Text preceding an anchor phrase.

	Returns
	-------
	str
		Matched candidate phrase, or an empty string.
	"""
	text = clean_candidate(text)
	if not text:
		return ""

	match = re.search(
		r"((?:[A-Za-z][A-Za-z\-]*\s+){0,5}"
		r"(?:equation|relation|law|Hamiltonian|Lagrangian|constraint|condition|"
		r"function|matrix|operator|state|model|expression|formula))$",
		text,
		re.I,
	)
	if not match:
		return ""
	return clean_candidate(match.group(1))


def source_text_slice(text: str, trigger_start: int, trigger_end: int) -> str:
	"""Create an audit excerpt around an anchor trigger.

	Parameters
	----------
	text : str
		Parsed document text.
	trigger_start : int
		Trigger start character offset.
	trigger_end : int
		Trigger end character offset.

	Returns
	-------
	str
		Normalized excerpt ending with the equation marker.
	"""
	left = text[:trigger_start].strip()
	right = text[trigger_start:trigger_end].strip()
	left_words = left.split()[-8:]
	return normalize_text(" ".join(left_words + [right, EQUATION_MARKER]))


def extract_equation_name(doc: spacy.tokens.Doc) -> tuple[str, dict]:
	"""Extract the most reliable equation name from parsed context.

	Anchor patterns are attempted first, followed by nearby noun chunks and
	dependency-context candidates.

	Parameters
	----------
	doc : spacy.tokens.Doc
		Parsed context containing an ``[EQUATION]`` marker.

	Returns
	-------
	tuple[str, dict]
		Extracted name and audit metadata. The name is empty when extraction
		fails.
	"""
	text = doc.text
	marker = marker_index(doc)
	if marker is None:
		return "", {
			"candidate": "",
			"confidence": "blank",
			"reason": "No [EQUATION] marker found",
			"strategy": "no_reliable_candidate",
			"trigger": "",
			"trigger_type": "",
			"source_text": "",
		}

	before_text = text[: doc[marker].idx]
	for trigger, pattern in ANCHOR_PATTERNS:
		matches = list(pattern.finditer(before_text))
		if not matches:
			continue

		match = matches[-1]
		prefix_doc = doc.char_span(0, match.start(), alignment_mode="contract")
		end_token_index = prefix_doc.end if prefix_doc is not None else marker
		candidate = best_noun_chunk_before(doc, end_token_index)
		if not candidate:
			candidate = fallback_phrase_before(before_text[: match.start()])
		if candidate and is_reliable_candidate(candidate, allow_generic=True):
			return candidate, {
				"candidate": candidate,
				"confidence": "high",
				"strategy": "anchor",
				"trigger": trigger,
				"trigger_type": "anchor_pattern",
				"source_text": source_text_slice(text, match.start(), match.end()),
			}

	fallback = pre_marker_noun_chunk_candidate(doc, marker)
	if fallback:
		return fallback

	fallback = dependency_context_candidate(doc, marker)
	if fallback:
		return fallback

	return "", {
		"candidate": "",
		"confidence": "blank",
		"reason": "No reliable SciSpaCy candidate found",
		"strategy": "no_reliable_candidate",
		"trigger": "",
		"trigger_type": "",
		"source_text": "",
	}


def meaning_audit(result: dict) -> dict:
	"""Convert extraction metadata to the output audit structure.

	Parameters
	----------
	result : dict
		Metadata produced by :func:`extract_equation_name`.

	Returns
	-------
	dict
		Nested ``meaning_extraction`` audit record.
	"""
	audit = {
		"method": "SciSpaCy/SciBERT dependency/noun-chunk extraction from surrounding_text.window",
		"candidate": result.get("candidate", ""),
		"confidence": result.get("confidence", "blank"),
		"strategy": result.get("strategy", ""),
	}
	if result.get("trigger"):
		audit["trigger"] = result["trigger"]
	if result.get("trigger_type"):
		audit["trigger_type"] = result["trigger_type"]
	if result.get("source_text"):
		audit["source_text"] = result["source_text"]
	if result.get("reason"):
		audit["reason"] = result["reason"]
	return {"meaning_extraction": audit}


def extract_meanings(
	input_file: Path = EQUATIONS_FILE,
	output_file: Path = EQUATION_MEANINGS_FILE,
) -> tuple[int, int, int]:
	"""Extract meanings for all equations and write the enriched dataset.

	Parameters
	----------
	input_file : pathlib.Path, default=EQUATIONS_FILE
		JSON equation dataset created by Step 3.
	output_file : pathlib.Path, default=EQUATION_MEANINGS_FILE
		Destination for the enriched JSON dataset.

	Returns
	-------
	tuple[int, int, int]
		Numbers of visited, filled, and blank equation entries.

	Raises
	------
	RuntimeError
		If the required SciSpaCy model cannot be loaded or process an input.
	"""
	nlp = load_spacy_model()
	data = json.loads(input_file.read_text(encoding="utf-8"))
	visited_count = 0
	filled_count = 0

	for paper_equations in data.values():
		for entry in paper_equations.values():
			visited_count += 1
			window = entry.get("surrounding_text", {}).get("window", "")
			doc, words_per_side = parse_window(nlp, window)
			candidate, result = extract_equation_name(doc)
			result["model_window_words_per_side"] = words_per_side
			if candidate:
				entry["meaning"] = candidate
				filled_count += 1

			audit_trail = entry.setdefault("audit-trail", [])
			audit_trail.append(meaning_audit(result))

	blank_count = visited_count - filled_count
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(
		json.dumps(data, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	return visited_count, filled_count, blank_count


def extract_meaning_main() -> int:
	"""Run equation-meaning extraction with the configured paths.

	Returns
	-------
	int
		Number of equations for which a meaning was extracted.
	"""
	visited_count, filled_count, blank_count = extract_meanings()
	print(f"Visited equations: {visited_count}")
	print(f"Meanings filled: {filled_count}")
	print(f"Meanings left blank: {blank_count}")
	print(f"Wrote equation meanings to {EQUATION_MEANINGS_FILE}")
	return filled_count


if __name__ == "__main__":
	try:
		extract_meaning_main()
	except RuntimeError as exc:
		print(exc, file=sys.stderr)
		sys.exit(1)
