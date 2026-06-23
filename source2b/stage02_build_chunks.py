"""Stage 2: Build BM25 retrieval chunks from structured documents."""

import sys
from pathlib import Path

from config import CHUNKS_DIR, DOCUMENTS_DIR

_SUBPKG = Path(__file__).parent.parent / "source2" / "chunk_builder"
_CONFLICTS = frozenset({"config", "models", "validation", "index"})


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    for n in _CONFLICTS:
        sys.modules.pop(n, None)
    try:
        from chunk_io import build_all_chunk_files

        paper_count, total_chunks = build_all_chunk_files(DOCUMENTS_DIR, CHUNKS_DIR)
        return {
            "paper_count": paper_count,
            "total_chunks": total_chunks,
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
