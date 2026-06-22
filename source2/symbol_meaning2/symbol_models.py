from dataclasses import asdict, dataclass, field
from typing import Any


RELATION_LABELS = (
	"DEFINES_COMPLETE_SYMBOL",
	"DEFINES_BASE",
	"QUALIFIES_SUBSCRIPT",
	"QUALIFIES_SUPERSCRIPT",
	"RELATED_NOT_DEFINITION",
	"NO_RELATION",
)


@dataclass(frozen=True)
class ParsedSymbol:
	original_latex: str
	base: str
	subscript: str | None
	superscript: str | None
	decorators: list[str]
	canonical: str
	equation: str
	aliases: list[str]

	@property
	def has_semantic_modifiers(self) -> bool:
		return bool(self.subscript or self.superscript or self.decorators)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(frozen=True)
class PhraseCandidate:
	phrase: str
	sentence: str
	result: dict
	source: str


@dataclass(frozen=True)
class RelationPrediction:
	phrase: str
	relation: str
	probabilities: dict[str, float]
	cross_encoder_score: float
	candidate: PhraseCandidate | None = None


@dataclass(frozen=True)
class SymbolMeaningRecord:
	canonical: str
	latex_forms: list[str]
	aliases: list[str]
	definition: str
	confidence: float
	strategy: str
	source_text: str
	source_chunk_id: str | None
	source_paragraph_ids: list[str] = field(default_factory=list)
	source_sentence_ids: list[str] = field(default_factory=list)
	retrieval_method: str | None = None
	retrieval_rank: int | None = None
	retrieval_score: float | None = None
	candidate_score: float = 0.0
	audit: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
