import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "postprocessing" / "equation_meanings"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "analysis" / "meaning_postprocessing_report.md"
DEFAULT_REJECTED_JSON_FILE = PROJECT_DIR / "analysis" / "rejected_equation_meanings.json"


def _load_records(input_dir: Path) -> list[tuple[str, dict[str, Any]]]:
	if not input_dir.is_dir():
		raise ValueError(f"Missing postprocessing directory: {input_dir}")
	records = []
	for input_file in sorted(input_dir.glob("*.json")):
		payload = json.loads(input_file.read_text(encoding="utf-8"))
		paper_id = payload.get("paper_id")
		if paper_id != input_file.stem:
			raise ValueError(f"Paper ID mismatch in {input_file}")
		for record in payload.get("equations", []):
			records.append((paper_id, record))
	return records


def build_report_data(
	input_dir: Path,
	examples_per_reason: int = 3,
) -> dict[str, Any]:
	records = _load_records(input_dir)
	strategies = Counter()
	rejection_records = Counter()
	rejection_candidates = Counter()
	examples: dict[str, list[dict[str, str]]] = defaultdict(list)
	phrase_lengths = []
	empty_count = 0
	flagged_count = 0
	changed_count = 0

	for paper_id, record in records:
		meaning = str(record.get("meaning", ""))
		audit = record.get("audit", {}).get("postprocessing", {})
		strategies[audit.get("strategy", "missing")] += 1
		changed_count += bool(audit.get("applied"))
		if meaning:
			phrase_lengths.append(len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", meaning)))
		else:
			empty_count += 1
		flagged = bool(audit.get("flagged")) or (
			not meaning and audit.get("strategy") == "no_reliable_phrase"
		)
		if not flagged:
			continue

		flagged_count += 1
		record_reasons = set()
		candidates = audit.get("candidates", [])
		if not candidates:
			record_reasons.add("no_candidates_generated")
		for candidate in candidates:
			candidate_reasons = candidate.get("reasons", [])
			rejection_candidates.update(candidate_reasons)
			record_reasons.update(candidate_reasons)
		for reason in record_reasons:
			rejection_records[reason] += 1
			if len(examples[reason]) < examples_per_reason:
				examples[reason].append({
					"paper_id": paper_id,
					"equation_id": str(record.get("equation_id", "")),
					"original_meaning": str(
						audit.get("original_meaning") or record.get("source_text", "")
					),
				})

	return {
		"summary": {
			"papers": len({paper_id for paper_id, _ in records}),
			"records": len(records),
			"nonempty": len(records) - empty_count,
			"empty": empty_count,
			"flagged": flagged_count,
			"changed": changed_count,
			"phrase_words": {
				"minimum": min(phrase_lengths, default=0),
				"median": statistics.median(phrase_lengths) if phrase_lengths else 0,
				"maximum": max(phrase_lengths, default=0),
			},
		},
		"strategies": dict(strategies.most_common()),
		"rejection_record_counts": dict(rejection_records.most_common()),
		"rejection_candidate_counts": dict(rejection_candidates.most_common()),
		"examples": dict(examples),
	}


def build_rejected_meanings(input_dir: Path) -> dict[str, Any]:
	rejected = []
	for paper_id, record in _load_records(input_dir):
		audit = record.get("audit", {}).get("postprocessing", {})
		flagged = bool(audit.get("flagged")) or (
			not record.get("meaning") and audit.get("strategy") == "no_reliable_phrase"
		)
		if not flagged:
			continue
		candidates = audit.get("candidates", [])
		reasons = sorted({
			reason
			for candidate in candidates
			for reason in candidate.get("reasons", [])
		})
		if not candidates:
			reasons.append("no_candidates_generated")
		rejected.append({
			"paper_id": paper_id,
			"equation_id": str(record.get("equation_id", "")),
			"equation": str(record.get("equation", "")),
			"original_meaning": str(
				audit.get("original_meaning") or record.get("source_text", "")
			),
			"source_text": str(record.get("source_text", "")),
			"rejection_reasons": reasons,
			"candidate_count": len(candidates),
			"candidates": candidates,
		})
	return {
		"summary": {
			"rejected_count": len(rejected),
			"note": "Reported validation cases; meanings are preserved in postprocessed data.",
			"source_directory": str(input_dir),
		},
		"rejected_meanings": rejected,
	}


def _escape_table(text: str) -> str:
	return " ".join(text.split()).replace("|", "\\|")


def render_markdown(report: dict[str, Any]) -> str:
	summary = report["summary"]
	lines = [
		"# Equation Meaning Postprocessing Report",
		"",
		"Generated from the postprocessed equation-meaning audit records.",
		"Flag counts overlap because one record may fail multiple checks. Flagged",
		"meanings are reported for review but preserved in postprocessed data.",
		"",
		"## Summary",
		"",
		"| Metric | Count |",
		"| --- | ---: |",
		f"| Papers | {summary['papers']} |",
		f"| Meaning records | {summary['records']} |",
		f"| Non-empty phrases | {summary['nonempty']} |",
		f"| Empty meanings | {summary['empty']} |",
		f"| Flagged meanings | {summary['flagged']} |",
		f"| Changed records | {summary['changed']} |",
		"",
		"Phrase length in natural-language words: "
		f"minimum {summary['phrase_words']['minimum']}, "
		f"median {summary['phrase_words']['median']}, "
		f"maximum {summary['phrase_words']['maximum']}.",
		"",
		"## Selection Strategies",
		"",
		"| Strategy | Records |",
		"| --- | ---: |",
	]
	for strategy, count in report["strategies"].items():
		lines.append(f"| `{strategy}` | {count} |")

	lines.extend([
		"",
		"## Flag Reasons",
		"",
		"`Records` counts unique flagged records affected by the reason. "
		"`Candidates` counts every candidate that failed validation.",
		"",
		"| Reason | Records | Candidates |",
		"| --- | ---: | ---: |",
	])
	for reason, count in report["rejection_record_counts"].items():
		candidate_count = report["rejection_candidate_counts"].get(reason, 0)
		lines.append(f"| `{reason}` | {count} | {candidate_count} |")

	lines.extend(["", "## Representative Flagged Cases", ""])
	for reason, entries in report["examples"].items():
		lines.extend([
			f"### `{reason}`",
			"",
			"| Paper | Equation | Original selected evidence |",
			"| --- | --- | --- |",
		])
		for entry in entries:
			meaning = _escape_table(entry["original_meaning"])
			if len(meaning) > 240:
				meaning = meaning[:237] + "..."
			lines.append(
				f"| `{entry['paper_id']}` | `{entry['equation_id']}` | {meaning} |"
			)
		lines.append("")
	return "\n".join(lines).rstrip() + "\n"


def export_report(
	input_dir: Path,
	output_file: Path,
	examples_per_reason: int = 3,
) -> dict[str, Any]:
	report = build_report_data(input_dir, examples_per_reason)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
	temporary_file.write_text(render_markdown(report), encoding="utf-8")
	temporary_file.replace(output_file)
	return report


def export_rejected_meanings(input_dir: Path, output_file: Path) -> dict[str, Any]:
	payload = build_rejected_meanings(input_dir)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)
	return payload


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Export equation-meaning postprocessing analysis",
	)
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
	parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
	parser.add_argument(
		"--rejected-json-file", type=Path, default=DEFAULT_REJECTED_JSON_FILE
	)
	parser.add_argument("--examples-per-reason", type=int, default=3)
	return parser


def main() -> None:
	arguments = _parser().parse_args()
	if arguments.examples_per_reason < 0:
		raise SystemExit("--examples-per-reason must be non-negative")
	report = export_report(
		arguments.input_dir,
		arguments.output_file,
		arguments.examples_per_reason,
	)
	rejected = export_rejected_meanings(
		arguments.input_dir,
		arguments.rejected_json_file,
	)
	summary = report["summary"]
	print(
		f"Wrote report for {summary['records']} meanings "
		f"({summary['empty']} empty) to {arguments.output_file}"
	)
	print(
		f"Wrote {rejected['summary']['rejected_count']} flagged meanings "
		f"to {arguments.rejected_json_file}"
	)


if __name__ == "__main__":
	main()
