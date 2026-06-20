from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
	chunk_id: str
	paper_id: str
	chunk_type: str
	text: str
	section_id: str | None
	section_title: str
	section_level: int | None
	section_kind: str | None
	paragraph_ids: list[str] = field(default_factory=list)
	sentence_ids: list[str] = field(default_factory=list)
	nearby_equation_ids: list[str] = field(default_factory=list)
	symbols: list[str] = field(default_factory=list)
	source: str = "html"

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
