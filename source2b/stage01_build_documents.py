"""Stage 1: Build structured documents from HTML papers."""

import sys
from pathlib import Path

from config import ANNOTATIONS_FILE, DOCUMENTS_DIR, EQUATIONS_FILE, HTML_DIR

_SUBPKG = Path(__file__).parent.parent / "source2" / "structure_builder"
_CONFLICTS = frozenset({"config", "models", "validation", "index"})


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    for n in _CONFLICTS:
        sys.modules.pop(n, None)
    try:
        from document_builder import build_corpus, write_papers

        corpus = build_corpus(ANNOTATIONS_FILE, EQUATIONS_FILE, HTML_DIR)
        write_papers(corpus, DOCUMENTS_DIR)
        report = corpus["build_report"]
        return {
            "paper_count": report["built_paper_count"],
            "total_equations": report["target_equation_count"],
            "total_sentences": report["sentence_count"],
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
