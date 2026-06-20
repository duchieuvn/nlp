import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import EQUATIONS_FILE, EQUATION_WINDOWS_FILE


EQUATION_MARKER = "[EQUATION]"
START_EQUATION_TOKEN = "<starteqn>"
END_EQUATION_TOKEN = "<endeqn>"
EQUATION_REFERENCE_TOKEN = "<eqref>"
OTHER_EQUATION_REFERENCE_TOKEN = "<other_eqref>"


def yaml_scalar(value: str) -> str:
	"""Return a YAML scalar that preserves backslashes literally."""
	while "\\\\" in value:
		value = value.replace("\\\\", "\\")
	return "'" + value.replace("'", "''") + "'"


def enrich_window(
	paper_id: str,
	equation_id: str,
	equation: str,
	window: str,
) -> str:
	marker_count = window.count(EQUATION_MARKER)
	if marker_count != 1:
		raise ValueError(
			f"Expected exactly one {EQUATION_MARKER} marker for paper {paper_id!r}, "
			f"equation {equation_id!r}; found {marker_count}"
		)

	self_reference = re.compile(
		rf"\bEq\.\s*\(\s*{re.escape(equation_id)}\s*\)",
		re.IGNORECASE,
	)
	window = self_reference.sub(EQUATION_REFERENCE_TOKEN, window)
	other_reference = re.compile(r"\bEqs?\.\s*\(\s*[^)]+?\s*\)", re.IGNORECASE)
	window = other_reference.sub(OTHER_EQUATION_REFERENCE_TOKEN, window)
	embedded_equation = f"{START_EQUATION_TOKEN} {equation} {END_EQUATION_TOKEN}"
	return window.replace(EQUATION_MARKER, embedded_equation, 1)


def extract_windows(data: dict[str, Any]) -> list[dict[str, str]]:
	records = []
	for paper_id, equations in data.items():
		if not isinstance(equations, dict):
			raise ValueError(f"Expected equations for paper {paper_id!r} to be an object")

		for equation_id, details in equations.items():
			try:
				equation = details["equation"]
				window = details["surrounding_text"]["window"]
			except (KeyError, TypeError) as error:
				raise ValueError(
					f"Missing equation or window for paper {paper_id!r}, equation {equation_id!r}"
				) from error

			if not isinstance(equation, str) or not isinstance(window, str):
				raise ValueError(
					f"Equation and window must be strings for paper {paper_id!r}, "
					f"equation {equation_id!r}"
				)

			records.append({
				"paper_id": paper_id,
				"equation_id": equation_id,
				"equation": equation,
				"window": enrich_window(paper_id, equation_id, equation, window),
			})

	return records


def write_yaml(records: list[dict[str, str]], output_file: Path) -> None:
	lines = []
	for record in records:
		lines.extend([
			f"- paper_id: {yaml_scalar(record['paper_id'])}",
			f"  equation_id: {yaml_scalar(record['equation_id'])}",
			f"  equation: {yaml_scalar(record['equation'])}",
			f"  window: {yaml_scalar(record['window'])}",
		])

	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(input_file: Path, output_file: Path) -> None:
	data = json.loads(input_file.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise ValueError("Top-level JSON value must be an object")

	records = extract_windows(data)
	write_yaml(records, output_file)
	print(f"Wrote {len(records)} equation windows to {output_file}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Export equation context windows from JSON to YAML."
	)
	parser.add_argument("--input", type=Path, default=EQUATIONS_FILE)
	parser.add_argument("--output", type=Path, default=EQUATION_WINDOWS_FILE)
	args = parser.parse_args()
	main(args.input, args.output)
