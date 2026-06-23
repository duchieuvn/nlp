"""Stage 7: Postprocess equation meanings — trim to short extractive phrases."""

import sys
from pathlib import Path

from config import EQUATION_MEANINGS_DIR, POSTPROCESSED_MEANINGS_DIR

_SUBPKG = Path(__file__).parent.parent / "source2" / "postprocessing"
_CONFLICTS: frozenset[str] = frozenset()


def run() -> dict:
    _saved = {n: sys.modules[n] for n in _CONFLICTS if n in sys.modules}
    _before = set(sys.modules)
    sys.path.insert(0, str(_SUBPKG))
    try:
        from postprocessing_io import postprocess_directory, summarize_directory

        POSTPROCESSED_MEANINGS_DIR.mkdir(parents=True, exist_ok=True)
        paper_count, total_records, total_changed = postprocess_directory(
            EQUATION_MEANINGS_DIR, POSTPROCESSED_MEANINGS_DIR
        )
        summary = summarize_directory(POSTPROCESSED_MEANINGS_DIR)
        return {
            "paper_count": paper_count,
            "total_records": total_records,
            "total_changed": total_changed,
            "total_nonempty": summary["nonempty"],
            "total_empty": summary["empty"],
            "total_flagged": summary["flagged"],
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
