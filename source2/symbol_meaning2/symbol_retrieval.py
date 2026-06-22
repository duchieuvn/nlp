from pathlib import Path
import sys

from .symbol_config import DEFINITION_CUES


RETRIEVAL_DIR = Path(__file__).parent.parent / "retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
	sys.path.insert(0, str(RETRIEVAL_DIR))

from retrieval_models import SearchQuery
from retrieval_service import RetrievalService


def load_retrieval_service(chunks_dir: Path) -> RetrievalService:
	return RetrievalService.from_directory(chunks_dir)


def build_symbol_query(symbol: dict) -> str:
	aliases = [alias.strip() for alias in symbol.get("aliases", []) if alias.strip()]
	return " ".join((*dict.fromkeys(aliases), *DEFINITION_CUES))


def retrieve_symbol_evidence(
	service, paper_id: str, equation_id: str, symbol: dict, top_k: int,
) -> tuple[str, list[dict]]:
	query_text = build_symbol_query(symbol)
	query = SearchQuery(
		text=query_text,
		paper_id=paper_id,
		chunk_types=["sentence", "paragraph", "equation_neighborhood"],
		top_k=top_k,
	)
	return query_text, [result.to_dict() for result in service.search(query, "bm25")]
