"""Stage 6: Extract per-symbol definitions via BM25 retrieval and scoring."""

import sys
from pathlib import Path

from config import CHUNKS_DIR, SYMBOL_MEANINGS_DIR, SYMBOL_MEANING_TOP_K, SYMBOLS_DIR

_SUBPKG = Path(__file__).parent.parent / "source2" / "symbol_meaning"
_CONFLICTS = frozenset({
    "symbol_config", "symbol_extractor", "symbol_io", "symbol_models",
    "candidates", "patterns", "spacy_fallback",
})


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    for n in _CONFLICTS:
        sys.modules.pop(n, None)
    try:
        from spacy_fallback import load_spacy_pipeline
        from symbol_io import extract_all_symbol_meanings
        from symbol_retrieval import load_retrieval_service

        SYMBOL_MEANINGS_DIR.mkdir(parents=True, exist_ok=True)
        service = load_retrieval_service(CHUNKS_DIR)
        nlp = load_spacy_pipeline()
        print(f"  spaCy dependency fallback: {'enabled' if nlp else 'unavailable'}")
        paper_count, total_symbols, total_definitions = extract_all_symbol_meanings(
            service,
            SYMBOLS_DIR,
            SYMBOL_MEANINGS_DIR,
            top_k=SYMBOL_MEANING_TOP_K,
            nlp=nlp,
        )
        return {
            "paper_count": paper_count,
            "total_symbols": total_symbols,
            "total_definitions_found": total_definitions,
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
