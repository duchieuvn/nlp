from dataclasses import dataclass
from typing import Any


def unique(values) -> list[str]:
	return list(dict.fromkeys(value for value in values if value))


@dataclass
class PaperIndex:
	paper: dict[str, Any]

	def __post_init__(self) -> None:
		self.sections = {
			section["section_id"]: section for section in self.paper["sections"]
		}
		self.paragraphs = {
			paragraph["paragraph_id"]: paragraph
			for section in self.paper["sections"]
			for paragraph in section["paragraphs"]
		}
		self.sentences = {
			sentence["sentence_id"]: sentence
			for paragraph in self.paragraphs.values()
			for sentence in paragraph["sentences"]
		}
		self.paragraph_sections = {
			paragraph["paragraph_id"]: section
			for section in self.paper["sections"]
			for paragraph in section["paragraphs"]
		}
		self.sentence_paragraphs = {
			sentence["sentence_id"]: paragraph
			for paragraph in self.paragraphs.values()
			for sentence in paragraph["sentences"]
		}
		self.references_by_sentence: dict[str, list[dict[str, Any]]] = {}
		for reference in self.paper["cross_references"]:
			self.references_by_sentence.setdefault(
				reference["source_sentence_id"], []
			).append(reference)

	def section_metadata(self, section_id: str | None) -> dict[str, Any]:
		section = self.sections.get(section_id)
		return {
			"section_id": section_id,
			"section_title": section["title"] if section else "",
			"section_level": section["level"] if section else None,
			"section_kind": section["kind"] if section else None,
		}

	def sentence_ids(self, paragraph_ids: list[str]) -> list[str]:
		return [
			sentence["sentence_id"]
			for paragraph_id in paragraph_ids
			if paragraph_id in self.paragraphs
			for sentence in self.paragraphs[paragraph_id]["sentences"]
		]

	def nearby_equation_ids(self, paragraph_ids: list[str]) -> list[str]:
		return unique(
			equation_id
			for paragraph_id in paragraph_ids
			if paragraph_id in self.paragraphs
			for equation_id in self.paragraphs[paragraph_id]["nearby_equation_ids"]
		)
