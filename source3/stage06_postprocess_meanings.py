"""Stage 6b: Postprocess extracted equation meanings with Source2 cleanup rules."""

import json
import shutil
import sys
from pathlib import Path

from config import MEANINGS_DIR, OUTPUT_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from source2.postprocessing.postprocessing_io import postprocess_payload, summarize_directory


def _write_json(payload: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output_file)


def _swap_directory(source_dir: Path, target_dir: Path) -> None:
    backup_dir = target_dir.with_name(f"{target_dir.name}.pre_postprocess_tmp")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    target_dir.replace(backup_dir)
    source_dir.replace(target_dir)
    shutil.rmtree(backup_dir)


def run() -> dict:
    """Postprocess Stage 6 meanings and replace MEANINGS_DIR atomically."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MEANINGS_DIR.exists():
        raise FileNotFoundError(f"Missing meanings directory: {MEANINGS_DIR}")

    meaning_files = sorted(MEANINGS_DIR.glob("*.json"))
    if not meaning_files:
        raise FileNotFoundError(f"No meaning files in {MEANINGS_DIR}")

    tmp_dir = OUTPUT_DIR / "meanings.postprocess_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    total_equations = 0
    total_changed = 0
    paper_results = []

    for input_file in meaning_files:
        payload = json.loads(input_file.read_text(encoding="utf-8"))
        updated, changed = postprocess_payload(payload)
        _write_json(updated, tmp_dir / input_file.name)

        equations = len(updated.get("equations", []))
        total_equations += equations
        total_changed += changed
        paper_results.append({
            "paper_id": input_file.stem,
            "equations": equations,
            "meanings_changed": changed,
        })

    summary = summarize_directory(tmp_dir)
    _swap_directory(tmp_dir, MEANINGS_DIR)

    return {
        "paper_count": len(meaning_files),
        "total_equations": total_equations,
        "total_meanings_changed": total_changed,
        "summary": summary,
        "papers": paper_results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
