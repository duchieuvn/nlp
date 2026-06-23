"""Stage 3: Extract symbols from equations and annotate chunks."""

import sys
from pathlib import Path

from config import CHUNKS_DIR, EQUATIONS_FILE, SYMBOLS_DIR

_SUBPKG = Path(__file__).parent.parent / "source2" / "symbol_extractor"
_CONFLICTS = frozenset({"symbol_config", "symbol_extractor", "symbol_io", "symbol_models"})


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    for n in _CONFLICTS:
        sys.modules.pop(n, None)
    try:
        from symbol_io import build_symbol_files

        paper_count, total_symbols, enriched_chunks = build_symbol_files(
            EQUATIONS_FILE, SYMBOLS_DIR, CHUNKS_DIR
        )
        return {
            "paper_count": paper_count,
            "total_symbols": total_symbols,
            "total_enriched_chunks": enriched_chunks,
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
