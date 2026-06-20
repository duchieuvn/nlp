from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolCandidate:
	canonical: str
	latex_forms: list[str]
	aliases: list[str]

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
