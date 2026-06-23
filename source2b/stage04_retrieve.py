"""Stage 4: BM25 retrieval — find candidate sentences for each equation."""

import sys
from pathlib import Path

from config import (
    CHUNKS_DIR,
    RETRIEVAL_METHOD,
    RETRIEVAL_RESULTS_DIR,
    RETRIEVAL_TOP_K,
    SYMBOLS_DIR,
)

_SUBPKG = Path(__file__).parent.parent / "source2" / "retrieval"
_CONFLICTS: frozenset[str] = frozenset()


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    try:
        from batch_search import search_all_papers
        from retrieval_service import RetrievalService

        RETRIEVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        service = RetrievalService.from_directory(CHUNKS_DIR)
        paper_count, total_queries, total_results = search_all_papers(
            service,
            SYMBOLS_DIR,
            RETRIEVAL_RESULTS_DIR,
            method=RETRIEVAL_METHOD,
            top_k=RETRIEVAL_TOP_K,
        )
        return {
            "paper_count": paper_count,
            "total_queries": total_queries,
            "total_results": total_results,
        }
    finally:
        try:
            sys.path.remove(str(_SUBPKG))
        except ValueError:
            pass
        for k in list(sys.modules):
            if k not in _before:
                sys.modules.pop(k, None)
        sys.modules.update(_saved)
