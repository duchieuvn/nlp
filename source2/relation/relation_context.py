from dataclasses import dataclass
import re

from relation_patterns import mentions_equation


TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
STOP_WORDS = {
	"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
	"in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
	"we", "where", "which", "with",
}


@dataclass(frozen=True)
class EvidenceSentence:
	text: str
	sentence_id: str
	sentence_ids: tuple[str, ...]
	paragraph_id: str
	section_id: str | None
	position: str


@dataclass(frozen=True)
class EquationContext:
	equation: dict
	sentences: list[EvidenceSentence]
	text: str


def build_equation_contexts(document: dict) -> dict[str, EquationContext]:
	paragraphs = {}
	for section in document["sections"]:
		for paragraph in section["paragraphs"]:
			paragraphs[paragraph["paragraph_id"]] = (section["section_id"], paragraph)
	previous_paragraph_owners = {
		equation.get("previous_paragraph_id"): equation["equation_id"]
		for equation in document["equations"]
		if equation.get("previous_paragraph_id")
	}

	contexts = {}
	for equation in document["equations"]:
		local_sentences = []
		for key in ("previous_paragraph_id", "next_paragraph_id"):
			paragraph_id = equation.get(key)
			if paragraph_id not in paragraphs:
				continue
			section_id, paragraph = paragraphs[paragraph_id]
			position = "before_equation" if key == "previous_paragraph_id" else "after_equation"
			if (
				key == "next_paragraph_id"
				and previous_paragraph_owners.get(paragraph_id) not in {None, equation["equation_id"]}
			):
				position = "introduces_other_equation"
			local_sentences.extend(
				_reconstructed_sentences(paragraph, section_id, position)
			)
		contexts[equation["equation_id"]] = EquationContext(
			equation=equation,
			sentences=list({item.sentence_id: item for item in local_sentences}.values()),
			text=" ".join(item.text for item in local_sentences),
		)
	return contexts


def _reconstructed_sentences(
	paragraph: dict,
	section_id: str,
	position: str,
) -> list[EvidenceSentence]:
	result = []
	buffer = []
	identifiers = []
	for sentence in paragraph["sentences"]:
		buffer.append(sentence["text"])
		identifiers.append(sentence["sentence_id"])
		text = " ".join(buffer)
		if text.count("(") > text.count(")"):
			continue
		result.append(EvidenceSentence(
			text=text,
			sentence_id=identifiers[0],
			sentence_ids=tuple(identifiers),
			paragraph_id=paragraph["paragraph_id"],
			section_id=section_id,
			position=position,
		))
		buffer = []
		identifiers = []
	if buffer:
		result.append(EvidenceSentence(
			text=" ".join(buffer),
			sentence_id=identifiers[0],
			sentence_ids=tuple(identifiers),
			paragraph_id=paragraph["paragraph_id"],
			section_id=section_id,
			position=position,
		))
	return result


def explicit_evidence(
	context: EquationContext,
	target_equation_id: str,
	cross_references: list[dict],
) -> list[EvidenceSentence]:
	eligible = [
		sentence for sentence in context.sentences
		if sentence.position != "introduces_other_equation"
	]
	local_ids = {
		sentence_id
		for sentence in eligible
		for sentence_id in sentence.sentence_ids
	}
	resolved_ids = {
		reference["source_sentence_id"]
		for reference in cross_references
		if target_equation_id in reference.get("target_equation_ids", [])
		and reference.get("source_sentence_id") in local_ids
	}
	return [
		sentence
		for sentence in eligible
		if bool(set(sentence.sentence_ids) & resolved_ids)
		or mentions_equation(sentence.text, target_equation_id)
	]


def token_set(text: str) -> set[str]:
	return {
		token.casefold()
		for token in TOKEN.findall(text)
		if token.casefold() not in STOP_WORDS and len(token) > 1
	}


def jaccard(left: set[str], right: set[str]) -> float:
	union = left | right
	return len(left & right) / len(union) if union else 0.0
