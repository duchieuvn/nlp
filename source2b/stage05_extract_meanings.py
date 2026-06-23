"""Stage 5: Extract equation meanings from BM25 candidates via MathBERT reranking."""

import sys
from pathlib import Path

from config import EQUATION_MEANINGS_DIR, EQUATIONS_FILE, RETRIEVAL_RESULTS_DIR

_SUBPKG = Path(__file__).parent.parent / "source2" / "eqn_meaning"
_CONFLICTS = frozenset({"candidates", "patterns"})


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    for n in _CONFLICTS:
        sys.modules.pop(n, None)
    try:
        from mathbert_reranker import MathBERTReranker
        from meaning_io import extract_all_meanings

        reranker = MathBERTReranker()
        print(f"  Loaded {reranker.model_name} on {reranker.device}")
        EQUATION_MEANINGS_DIR.mkdir(parents=True, exist_ok=True)
        paper_count, total_equations, total_meanings = extract_all_meanings(
            RETRIEVAL_RESULTS_DIR,
            EQUATIONS_FILE,
            EQUATION_MEANINGS_DIR,
            reranker=reranker,
        )
        return {
            "paper_count": paper_count,
            "total_equations": total_equations,
            "total_meanings_found": total_meanings,
            "total_meanings_empty": total_equations - total_meanings,
        }
    finally:
        try:
            sys.path.remove(str(_SUBPKG))
        except ValueError:
            pass
        for k in list(sys.modules):
            if k not in _before:
                sys.modules.pop(k, None)
        for n in _CONFLICTS:
            sys.modules.pop(n, None)
        sys.modules.update(_saved)
