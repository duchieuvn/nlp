import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SYMBOL_MEANING_DIR = PROJECT_DIR / "source2" / "symbol_meaning"
RETRIEVAL_DIR = PROJECT_DIR / "source2" / "retrieval"
for module_dir in (SYMBOL_MEANING_DIR, RETRIEVAL_DIR):
	if str(module_dir) not in sys.path:
		sys.path.insert(0, str(module_dir))

from candidates import collect_candidates
from patterns import extract_definition, mentions_alias
from symbol_config import CHUNKS_DIR, SYMBOL_MEANINGS_DIR, TOP_K
from symbol_retrieval import load_retrieval_service, retrieve_symbol_evidence


DEFAULT_OUTPUT_FILE = PROJECT_DIR / "analysis" / "rejected_symbol_meanings.json"
DEFAULT_REPORT_FILE = PROJECT_DIR / "analysis" / "symbol_meaning_rejection_report.md"


def _load_records(input_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
	if not input_dir.is_dir():
		raise ValueError(f"Missing symbol-meaning directory: {input_dir}")
	records = []
	for input_file in sorted(input_dir.glob("*.json")):
		payload = json.loads(input_file.read_text(encoding="utf-8"))
		paper_id = payload.get("paper_id")
		if paper_id != input_file.stem:
			raise ValueError(f"Paper ID mismatch in {input_file}")
		for equation in payload.get("equations", []):
			equation_id = str(equation.get("equation_id", ""))
			for symbol in equation.get("symbols", []):
				records.append((paper_id, equation_id, {
					"equation": str(equation.get("latex", "")),
					**symbol,
				}))
	return records


def _candidate_record(candidate, aliases: list[str]) -> dict[str, Any]:
	alias_mentioned = mentions_alias(candidate.text, aliases)
	definition = ""
	strategy = "no_pattern"
	matched_alias = ""
	if alias_mentioned:
		definition, strategy, matched_alias = extract_definition(
			candidate.text, aliases
		)
	reasons = []
	if not alias_mentioned:
		reasons.append("alias_not_mentioned")
	elif not definition:
		reasons.append("no_supported_definition_pattern")
	else:
		reasons.append("definition_found_but_record_empty")
	result = candidate.result
	return {
		"chunk_id": str(result.get("chunk_id", "")),
		"chunk_type": str(result.get("chunk_type", "")),
		"rank": result.get("rank"),
		"score": result.get("score"),
		"text": candidate.text,
		"alias_mentioned": alias_mentioned,
		"matched_alias": matched_alias,
		"extracted_definition": definition,
		"strategy": strategy,
		"accepted": False,
		"rejection_reasons": reasons,
	}


def build_rejected_symbol_meanings(
	input_dir: Path,
	service,
	top_k: int = TOP_K,
) -> dict[str, Any]:
	rejected = []
	reason_counts: Counter[str] = Counter()
	records = _load_records(input_dir)
	for paper_id, equation_id, symbol in records:
		if str(symbol.get("definition", "")).strip():
			continue
		aliases = list(dict.fromkeys(symbol.get("aliases", [])))
		query, results = retrieve_symbol_evidence(
			service, paper_id, equation_id, symbol, top_k
		)
		candidates = [
			_candidate_record(candidate, aliases)
			for candidate in collect_candidates(results)
		]
		if not candidates:
			reasons = ["no_candidates_retrieved"]
		elif not any(candidate["alias_mentioned"] for candidate in candidates):
			reasons = ["no_retrieved_alias"]
		elif not any(candidate["extracted_definition"] for candidate in candidates):
			reasons = ["no_supported_definition_pattern"]
		else:
			reasons = ["definition_found_but_record_empty"]
		reason_counts.update(reasons)
		rejected.append({
			"paper_id": paper_id,
			"equation_id": equation_id,
			"equation": symbol["equation"],
			"canonical": str(symbol.get("canonical", "")),
			"latex_forms": list(symbol.get("latex_forms", [])),
			"aliases": aliases,
			"query": query,
			"rejection_reasons": reasons,
			"candidate_count": len(candidates),
			"candidates": candidates,
		})
	return {
		"summary": {
			"papers": len({paper_id for paper_id, _, _ in records}),
			"symbol_count": len(records),
			"rejected_count": len(rejected),
			"defined_count": len(records) - len(rejected),
			"empty_percentage": round(
				100 * len(rejected) / len(records), 2
			) if records else 0.0,
			"rejection_reason_counts": dict(reason_counts.most_common()),
			"retrieval_method": "bm25",
			"top_k": top_k,
			"source_directory": str(input_dir),
		},
		"rejected_symbol_meanings": rejected,
	}


def _escape_table(text: str) -> str:
	return " ".join(text.split()).replace("|", "\\|")


def render_markdown(payload: dict[str, Any], examples_per_reason: int = 3) -> str:
	summary = payload["summary"]
	lines = [
		"# Symbol Meaning Rejection Report",
		"",
		"Generated from empty symbol-meaning records and reconstructed BM25 evidence.",
		"A rejected symbol has no reliable extractive definition; the symbol itself",
		"is still preserved in the final dataset.",
		"",
		"## Summary",
		"",
		"| Metric | Count |",
		"| --- | ---: |",
		f"| Papers | {summary['papers']} |",
		f"| Extracted symbols | {summary['symbol_count']} |",
		f"| Defined symbols | {summary['defined_count']} |",
		f"| Empty definitions | {summary['rejected_count']} |",
		f"| Empty percentage | {summary['empty_percentage']:.2f}% |",
		"",
		f"Retrieval uses `{summary['retrieval_method']}` with top-k "
		f"`{summary['top_k']}` candidates.",
		"",
		"## Rejection Reasons",
		"",
		"| Reason | Symbols | Percentage of empty definitions |",
		"| --- | ---: | ---: |",
	]
	for reason, count in summary["rejection_reason_counts"].items():
		percentage = 100 * count / summary["rejected_count"] if summary["rejected_count"] else 0
		lines.append(f"| `{reason}` | {count} | {percentage:.2f}% |")

	lines.extend(["", "## Representative Rejected Cases", ""])
	for reason in summary["rejection_reason_counts"]:
		matching = [
			record
			for record in payload["rejected_symbol_meanings"]
			if reason in record["rejection_reasons"]
		][:examples_per_reason]
		lines.extend([
			f"### `{reason}`",
			"",
			"| Paper | Equation | Symbol | Best retrieved evidence |",
			"| --- | --- | --- | --- |",
		])
		for record in matching:
			alias_candidates = [
				candidate
				for candidate in record["candidates"]
				if candidate["alias_mentioned"]
			]
			candidate = (alias_candidates or record["candidates"] or [{}])[0]
			evidence = _escape_table(str(candidate.get("text", "")))
			if len(evidence) > 240:
				evidence = evidence[:237] + "..."
			lines.append(
				f"| `{record['paper_id']}` | `{record['equation_id']}` | "
				f"`{record['canonical']}` | {evidence} |"
			)
		lines.append("")
	return "\n".join(lines).rstrip() + "\n"


def export_markdown_report(payload: dict[str, Any], output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
	temporary_file.write_text(render_markdown(payload), encoding="utf-8")
	temporary_file.replace(output_file)


def export_rejected_symbol_meanings(
	input_dir: Path,
	chunks_dir: Path,
	output_file: Path,
	top_k: int = TOP_K,
) -> dict[str, Any]:
	service = load_retrieval_service(chunks_dir)
	payload = build_rejected_symbol_meanings(input_dir, service, top_k)
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
		description="Export empty symbol meanings and their rejected evidence",
	)
	parser.add_argument("--input-dir", type=Path, default=SYMBOL_MEANINGS_DIR)
	parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
	parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
	parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
	parser.add_argument("--top-k", type=int, default=TOP_K)
	return parser


def main() -> None:
	arguments = _parser().parse_args()
	if arguments.top_k < 1:
		raise SystemExit("--top-k must be positive")
	payload = export_rejected_symbol_meanings(
		arguments.input_dir,
		arguments.chunks_dir,
		arguments.output_file,
		arguments.top_k,
	)
	export_markdown_report(payload, arguments.report_file)
	summary = payload["summary"]
	print(
		f"Wrote {summary['rejected_count']} rejected symbol meanings "
		f"out of {summary['symbol_count']} symbols to {arguments.output_file}"
	)
	print(f"Wrote Markdown summary to {arguments.report_file}")


if __name__ == "__main__":
	main()
