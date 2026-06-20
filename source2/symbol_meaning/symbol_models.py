from dataclasses import asdict, dataclass, field
from typing import Any


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
