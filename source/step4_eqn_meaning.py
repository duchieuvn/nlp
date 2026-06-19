from pathlib import Path
from dataclasses import dataclass
import json
import re
import sys

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
	return re.sub(r"\s+", " ", text).strip()


def clean_model_text(text: str) -> str:
	text = re.sub(r"\\[A-Za-z]+", " ", text)
	text = re.sub(r"[{}_^]", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def model_input_window(window: str, words_per_side: int) -> str:
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
	message = str(exc)
	return "512" in message and "tensor" in message


def parse_window(nlp, window: str):
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
	text = normalize_text(text)
	text = re.sub(r"^[,;:.\s]+|[,;:.\s]+$", "", text)
	words = text.split()
	while words and words[0].lower() in LEADING_DROP_WORDS:
		words.pop(0)
	return " ".join(words)


def candidate_words(text: str) -> list[str]:
	return [word.lower() for word in re.findall(r"[A-Za-z]+", text)]


def marker_index(doc) -> int | None:
	for token in doc:
		if token.text == EQUATION_MARKER:
			return token.i
	return None


def has_science_head(text: str) -> bool:
	words = candidate_words(text)
	return any(word in SCIENCE_HEAD_WORDS for word in words)


def is_generic_candidate(text: str) -> bool:
	words = candidate_words(text)
	return len(words) == 1 and words[0] in GENERIC_CANDIDATES


def is_reference_candidate(text: str) -> bool:
	return bool(re.search(r"\b(?:eq|fig|sec|app|ref)s?\.?\s*\(?\s*\d+", text, re.I))


def is_symbol_heavy_candidate(text: str) -> bool:
	if re.search(r"[()[\]{}]", text):
		return True
	compact = re.sub(r"\s+", "", text)
	if not compact:
		return True
	alpha_count = len(re.findall(r"[A-Za-z]", compact))
	symbol_count = len(compact) - alpha_count
	return alpha_count < 3 or (symbol_count > alpha_count and alpha_count < 8)


def is_reliable_candidate(text: str, allow_generic: bool = False) -> bool:
	words = candidate_words(text)
	if not words or len(words) > 12:
		return False
	if is_reference_candidate(text) or is_symbol_heavy_candidate(text):
		return False
	if is_generic_candidate(text) and not allow_generic:
		return False
	return True


def noun_chunk_score(chunk_text: str) -> tuple[int, int]:
	words = candidate_words(chunk_text)
	has_science_head = any(word in SCIENCE_HEAD_WORDS for word in words)
	return (1 if has_science_head else 0, len(words))


def is_anchor_start(doc, token_index: int, max_end_token_index: int) -> bool:
	text = doc[token_index:max_end_token_index].text
	return any(pattern.match(text) for _, pattern in ANCHOR_PATTERNS)


def expansion_stop_index(doc, start_token_index: int, max_end_token_index: int) -> int:
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


def expand_noun_chunk(chunk, doc, max_end_token_index: int) -> tuple[int, int, str]:
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
	word_count = len(candidate_words(candidate))
	return (
		1 if has_science_head(candidate) else 0,
		1 if same_sentence else 0,
		1 if expanded_end > chunk_end else 0,
		min(word_count, 8),
		-(reference_token_index - expanded_end),
	)


def best_noun_chunk_before(doc, end_token_index: int) -> str:
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


def sentence_bounds_for_marker(doc, marker: int) -> tuple[int, int]:
	try:
		sentence = doc[marker].sent
		return sentence.start, sentence.end
	except ValueError:
		return 0, len(doc)


def source_text_for_candidate(doc, start: int, end: int, marker: int) -> str:
	left = doc[max(0, start - 8) : start].text
	candidate = doc[start:end].text
	right = doc[end : min(len(doc), marker)].text
	return normalize_text(" ".join([left, candidate, right, EQUATION_MARKER]))


def scored_noun_chunks_before(doc, marker: int) -> list[CandidateSpan]:
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


def sentence_has_context_verb(doc, marker: int) -> bool:
	sentence_start, _ = sentence_bounds_for_marker(doc, marker)
	for token in doc[sentence_start:marker]:
		token_text = token.text.lower()
		if token.lemma_.lower() in CONTEXT_VERB_LEMMAS:
			return True
		if token_text in CONTEXT_VERB_FORMS:
			return True
	return False


def pre_marker_noun_chunk_candidate(doc, marker: int) -> tuple[str, dict] | None:
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


def dependency_context_candidate(doc, marker: int) -> tuple[str, dict] | None:
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
	left = text[:trigger_start].strip()
	right = text[trigger_start:trigger_end].strip()
	left_words = left.split()[-8:]
	return normalize_text(" ".join(left_words + [right, EQUATION_MARKER]))


def extract_equation_name(doc) -> tuple[str, dict]:
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
