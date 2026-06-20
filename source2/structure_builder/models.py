from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class SentenceRecord:
	sentence_id: str
	order: int
	text: str
	start: int
	end: int
	cross_reference_ids: list[str] = field(default_factory=list)


@dataclass
class ParagraphRecord:
	paragraph_id: str
	order: int
	document_order: int
	text: str
	sentences: list[SentenceRecord] = field(default_factory=list)
	nearby_equation_ids: list[str] = field(default_factory=list)
	cross_reference_ids: list[str] = field(default_factory=list)


@dataclass
class SectionRecord:
	section_id: str
	parent_section_id: str | None
	order: int
	level: int
	kind: str
	title: str
	synthetic: bool
	paragraphs: list[ParagraphRecord] = field(default_factory=list)
	equation_ids: list[str] = field(default_factory=list)


@dataclass
class EquationRecord:
	equation_id: str
	latex: str
	section_id: str | None
	document_order: int | None
	anchor_id: str | None
	annotation_ids: list[str]
	match_method: str
	previous_paragraph_id: str | None
	next_paragraph_id: str | None
	legacy_context_before: str
	legacy_context_after: str


@dataclass
class CrossReferenceRecord:
	reference_id: str
	raw_text: str
	reference_type: str
	source_section_id: str
	source_paragraph_id: str
	source_sentence_id: str
	paragraph_start: int
	paragraph_end: int
	sentence_start: int
	sentence_end: int
	target_equation_ids: list[str]
	unresolved_labels: list[str]


@dataclass
class PaperDocument:
	paper_id: str
	title: str
	html_source: str
	source_status: str
	sections: list[SectionRecord]
	equations: list[EquationRecord]
	cross_references: list[CrossReferenceRecord]

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
