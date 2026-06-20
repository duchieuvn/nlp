from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkDocument:
	chunk_id: str
	paper_id: str
	chunk_type: str
	text: str
	section_id: str | None
	section_title: str
	paragraph_ids: list[str]
	sentence_ids: list[str]
	nearby_equation_ids: list[str]
	symbols: list[str]
	source: str

	@classmethod
	def from_dict(cls, value: dict[str, Any]) -> "ChunkDocument":
		return cls(
			chunk_id=value["chunk_id"],
			paper_id=value["paper_id"],
			chunk_type=value["chunk_type"],
			text=value["text"],
			section_id=value.get("section_id"),
			section_title=value.get("section_title", ""),
			paragraph_ids=list(value.get("paragraph_ids", [])),
			sentence_ids=list(value.get("sentence_ids", [])),
			nearby_equation_ids=list(value.get("nearby_equation_ids", [])),
			symbols=list(value.get("symbols", [])),
			source=value.get("source", "html"),
		)


@dataclass(frozen=True)
class SearchQuery:
	text: str
	paper_id: str
	section_ids: list[str] | None = None
	chunk_types: list[str] | None = None
	equation_ids: list[str] | None = None
	symbols: list[str] | None = None
	top_k: int = 10

	def __post_init__(self) -> None:
		if not self.text.strip():
			raise ValueError("Search query text must not be empty")
		if not self.paper_id:
			raise ValueError("Search query requires a paper ID")
		if self.top_k < 1:
			raise ValueError("top_k must be at least 1")


@dataclass(frozen=True)
class SearchResult:
	rank: int
	chunk_id: str
	score: float
	method: str
	chunk_type: str
	text: str
	paper_id: str
	section_id: str | None
	section_title: str
	paragraph_ids: list[str] = field(default_factory=list)
	sentence_ids: list[str] = field(default_factory=list)
	nearby_equation_ids: list[str] = field(default_factory=list)
	symbols: list[str] = field(default_factory=list)
	source: str = "html"

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
