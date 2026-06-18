from pathlib import Path
import json
import re

from bs4 import BeautifulSoup


TARGET_MIN_EQUATIONS = 350


def read_paper_list(paper_list_file: Path) -> list[str]:
	paper_ids = []

	for line in paper_list_file.read_text(encoding="utf-8").splitlines():
		paper_id = line.strip()
		if not paper_id:
			continue
		if paper_id.startswith("arXiv:"):
			paper_id = paper_id.removeprefix("arXiv:")
		paper_ids.append(paper_id)

	return paper_ids


def normalize_latex(latex: str) -> str:
	latex = latex.replace("%\n", "")
	latex = latex.replace("\n", " ")
	latex = re.sub(r"^\s*\\displaystyle\s*", "", latex)
	latex = re.sub(r"\\begin\{(?:aligned|split)\}", "", latex)
	latex = re.sub(r"\\end\{(?:aligned|split)\}", "", latex)
	latex = re.sub(r"\s+", " ", latex)
	return latex.strip()


def extract_annotation_latex(annotation_html: str) -> str:
	soup = BeautifulSoup(annotation_html, "html.parser")
	return normalize_latex(soup.get_text())


def format_equation(annotation_html_list: list[str]) -> str:
	parts = [
		extract_annotation_latex(annotation_html)
		for annotation_html in annotation_html_list
	]
	parts = [part for part in parts if part]
	if not parts:
		return ""

	return " ".join(parts).rstrip(".,")


def make_entry(equation_key: str, equation: str, found_latex: bool) -> dict:
	audit_trail = {
		"extract_eq": f"Found numbered equation ({equation_key}) in 2_annotations.json.",
		"extract_latex": "Extracted LaTeX from annotation encoding application/x-tex."
		if found_latex
		else "No annotation encoding application/x-tex was found for this equation.",
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


def extract_equations(
	input_file: Path,
	output_file: Path,
	paper_list_file: Path,
	target_min_equations: int = TARGET_MIN_EQUATIONS,
) -> tuple[int, int, int]:
	annotations_by_paper = json.loads(input_file.read_text(encoding="utf-8"))
	paper_ids = read_paper_list(paper_list_file)
	equations_by_paper = {}
	seen_papers = set()
	converted_count = 0
	empty_paper_count = 0

	for paper_id in paper_ids:
		if paper_id in seen_papers:
			continue
		seen_papers.add(paper_id)

		paper_equations = {}
		annotations_by_number = annotations_by_paper.get(paper_id, {})

		for key, annotation_html_list in annotations_by_number.items():
			equation = format_equation(annotation_html_list)
			entry = make_entry(key, equation, bool(annotation_html_list))
			paper_equations[key] = entry
			converted_count += 1

		equations_by_paper[paper_id] = paper_equations
		if not paper_equations:
			empty_paper_count += 1

		if converted_count >= target_min_equations:
			break

	output_file.write_text(
		json.dumps(equations_by_paper, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	return converted_count, len(equations_by_paper), empty_paper_count


def main() -> None:
	from config import DATA_DIR

	configured_data_dir = Path(DATA_DIR)
	if configured_data_dir.is_absolute():
		data_dir = configured_data_dir
	else:
		data_dir = Path(__file__).resolve().parent / configured_data_dir
	input_file = data_dir / "2_annotations.json"
	output_file = data_dir / "3_equations.json"
	paper_list_file = data_dir / "paper_list_46.txt"

	equation_count, paper_count, empty_paper_count = extract_equations(
		input_file,
		output_file,
		paper_list_file,
	)
	print(f"Wrote {equation_count} equation entries to {output_file}")
	print(f"Output has {paper_count} papers and {equation_count} equations.")
	print(f"Empty/not found papers: {empty_paper_count}")


if __name__ == "__main__":
	main()
