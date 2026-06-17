from pathlib import Path
import json
import re

from bs4 import BeautifulSoup


TAG_SELECTOR = "span.ltx_tag_equation, span.ltx_tag_equationgroup"
TEX_SELECTOR = 'annotation[encoding="application/x-tex"]'
MAX_EQUATIONS_PER_PAPER = 7


def normalize_latex(latex: str) -> str:
	latex = latex.replace("%\n", "")
	latex = latex.replace("\n", " ")
	latex = re.sub(r"^\s*\\displaystyle\s*", "", latex)
	latex = re.sub(r"\\begin\{(?:aligned|split)\}", "", latex)
	latex = re.sub(r"\\end\{(?:aligned|split)\}", "", latex)
	latex = re.sub(r"\s+", " ", latex)
	return latex.strip()


def equation_key(tag_text: str) -> str:
	tag_text = tag_text.strip()
	if tag_text.startswith("(") and tag_text.endswith(")"):
		return tag_text[1:-1]
	return tag_text


def extract_rows(tbody_html: str) -> tuple[str | None, str, list[list[str]]]:
	soup = BeautifulSoup(tbody_html, "html.parser")
	tag = soup.select_one(TAG_SELECTOR)
	tag_text = tag.get_text(strip=True) if tag else ""
	key = equation_key(tag_text) if tag_text else ""

	rows = []
	for row in soup.select("tr"):
		row_latex = [
			normalize_latex(annotation.get_text())
			for annotation in row.select(TEX_SELECTOR)
		]
		if row_latex:
			rows.append(row_latex)

	if not rows:
		rows = [
			[normalize_latex(annotation.get_text())]
			for annotation in soup.select(TEX_SELECTOR)
		]

	return tag_text or None, key, rows


def format_equation(rows: list[list[str]]) -> str:
	if not rows:
		return ""

	lines = [" ".join(part for part in row if part).strip() for row in rows]
	lines = [line for line in lines if line]
	if not lines:
		return ""

	return " ".join(lines).rstrip(".,")


def make_entry(tag_text: str | None, equation: str, found_latex: bool) -> dict:
	display_tag = tag_text or "(unknown)"
	audit_trail = {
		"extract_eq": f"Found numbered equation {display_tag} in html_equations.json.",
		"extract_latex": "Extracted LaTeX from annotation encoding application/x-tex."
		if found_latex
		else "No annotation encoding application/x-tex was found for this equation block.",
		"normalize_latex": "Applied light cleanup for simple LaTeX output."
		if found_latex
		else "Skipped LaTeX cleanup because no source annotation was available.",
	}

	return {
		"equation": equation,
		"meaning": "",
		"symbols": {},
		"relations": {},
		"audit-trail": audit_trail,
	}


def unique_missing_key(index: int) -> str:
	return f"unknown-{index}"


def dump_without_backslash_escaping(data: dict) -> str:
	return json.dumps(data, indent=2, ensure_ascii=False).replace("\\\\", "\\")


def extract_equations(input_file: Path, output_file: Path) -> int:
	html_equations = json.loads(input_file.read_text(encoding="utf-8"))
	equations_by_paper = {}
	converted_count = 0
	missing_tag_count = 0

	for paper_id, tbody_blocks in html_equations.items():
		paper_equations = {}

		for tbody_html in tbody_blocks[:MAX_EQUATIONS_PER_PAPER]:
			tag_text, key, rows = extract_rows(tbody_html)
			if not key:
				missing_tag_count += 1
				key = unique_missing_key(missing_tag_count)

			equation = format_equation(rows)
			entry = make_entry(tag_text, equation, bool(rows))

			if key in paper_equations:
				paper_equations[key]["audit-trail"]["duplicate_warning"] = (
					f"Duplicate equation key {key} was found later in html_equations.json; "
					"kept the first occurrence and ignored the duplicate."
				)
				continue

			paper_equations[key] = entry
			converted_count += 1

		equations_by_paper[paper_id] = paper_equations

	output_file.write_text(
		dump_without_backslash_escaping(equations_by_paper) + "\n",
		encoding="utf-8",
	)
	return converted_count


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	input_file = script_dir / "html_equations.json"
	output_file = script_dir / "equations.json"

	count = extract_equations(input_file, output_file)
	print(f"Wrote {count} equation entries to {output_file}")


if __name__ == "__main__":
	main()
