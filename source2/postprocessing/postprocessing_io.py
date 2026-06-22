import json
from collections import Counter
from pathlib import Path
import statistics
from typing import Any

try:
	from .meaning_cleaner import WORD, postprocess_record
except ImportError:
	from meaning_cleaner import WORD, postprocess_record


def postprocess_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
	updated = dict(payload)
	records = [postprocess_record(record) for record in payload.get("equations", [])]
	updated["equations"] = records
	changed = sum(
		bool(record["audit"]["postprocessing"]["applied"])
		for record in records
	)
	return updated, changed


def _write_json(payload: dict[str, Any], output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)


def postprocess_file(input_file: Path, output_file: Path) -> tuple[int, int]:
	payload = json.loads(input_file.read_text(encoding="utf-8"))
	updated, changed = postprocess_payload(payload)
	_write_json(updated, output_file)
	return len(updated.get("equations", [])), changed


def postprocess_directory(
	input_dir: Path,
	output_dir: Path,
	paper_ids: set[str] | None = None,
) -> tuple[int, int, int]:
	if input_dir.resolve() == output_dir.resolve():
		raise ValueError("Postprocessing output directory must differ from input")
	files = sorted(input_dir.glob("*.json"))
	if paper_ids is not None:
		files = [input_file for input_file in files if input_file.stem in paper_ids]
		missing = paper_ids - {input_file.stem for input_file in files}
		if missing:
			raise ValueError(f"Unknown paper IDs: {', '.join(sorted(missing))}")

	record_count = 0
	changed_count = 0
	for input_file in files:
		records, changed = postprocess_file(input_file, output_dir / input_file.name)
		record_count += records
		changed_count += changed
	return len(files), record_count, changed_count


def summarize_directory(
	output_dir: Path,
	paper_ids: set[str] | None = None,
) -> dict[str, Any]:
	strategies = Counter()
	lengths = []
	record_count = 0
	empty_count = 0
	shortened_count = 0
	validation_failures = 0
	flagged_count = 0
	for output_file in sorted(output_dir.glob("*.json")):
		if paper_ids is not None and output_file.stem not in paper_ids:
			continue
		payload = json.loads(output_file.read_text(encoding="utf-8"))
		for record in payload.get("equations", []):
			record_count += 1
			meaning = record.get("meaning", "")
			postprocessing = record.get("audit", {}).get("postprocessing", {})
			strategies[postprocessing.get("strategy", "missing")] += 1
			flagged_count += bool(postprocessing.get("flagged"))
			if meaning:
				lengths.append(len(WORD.findall(meaning)))
				if not postprocessing.get("extractive"):
					validation_failures += 1
			else:
				empty_count += 1
			if postprocessing.get("applied") and meaning:
				shortened_count += 1
	return {
		"records": record_count,
		"nonempty": len(lengths),
		"empty": empty_count,
		"shortened": shortened_count,
		"strategies": dict(sorted(strategies.items())),
		"phrase_words": {
			"minimum": min(lengths, default=0),
			"median": statistics.median(lengths) if lengths else 0,
			"maximum": max(lengths, default=0),
		},
		"validation_failures": validation_failures,
		"flagged": flagged_count,
	}
