from pathlib import Path
from copy import copy
import json
import re

from bs4 import BeautifulSoup
from config import ANNOTATIONS_FILE, EQUATIONS_FILE, HTML_DIR, LIST_FILE


TARGET_MIN_EQUATIONS = 350
CONTEXT_SENTENCE_LIMIT = 3
EQUATION_MARKER = "[EQUATION]"
SENTENCE_ABBREVIATIONS = (
	"Eq.",
	"Eqs.",
	"Fig.",
	"Figs.",
	"Sec.",
	"Secs.",
	"Ref.",
	"Refs.",
	"App.",
)
ABBREVIATION_PERIOD = "<prd>"


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


def extract_annotation_ids(annotation_html_list: list[str]) -> list[str]:
	annotation_ids = []

	for annotation_html in annotation_html_list:
		soup = BeautifulSoup(annotation_html, "html.parser")
		annotation = soup.find("annotation")
		if annotation and annotation.get("id"):
			annotation_ids.append(annotation["id"])

	return annotation_ids


def format_equation(annotation_html_list: list[str]) -> str:
	parts = [
		extract_annotation_latex(annotation_html)
		for annotation_html in annotation_html_list
	]
	parts = [part for part in parts if part]
	if not parts:
		return ""

	return " ".join(parts).rstrip(".,")


def classes_contain(tag, class_name: str) -> bool:
	return class_name in (tag.get("class") or [])


def is_equation_table(tag) -> bool:
	return tag.name == "table" and (
		classes_contain(tag, "ltx_equation")
		or classes_contain(tag, "ltx_equationgroup")
	)


def is_context_boundary(tag) -> bool:
	return is_equation_table(tag)


def find_equation_anchor(soup: BeautifulSoup, annotation_ids: list[str]):
	for annotation_id in annotation_ids:
		annotation = soup.find(id=annotation_id)
		if not annotation:
			continue

		anchor = annotation.find_parent(is_equation_table)
		if anchor:
			return anchor, annotation_id

		for parent_name in ("tbody", "tr", "math"):
			anchor = annotation.find_parent(parent_name)
			if anchor:
				return anchor, annotation_id

	return None, None


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def protect_sentence_abbreviations(text: str) -> str:
	for abbreviation in SENTENCE_ABBREVIATIONS:
		text = text.replace(abbreviation, abbreviation.replace(".", ABBREVIATION_PERIOD))
	return text


def restore_sentence_abbreviations(text: str) -> str:
	return text.replace(ABBREVIATION_PERIOD, ".")


def clean_text_node(node) -> str:
	node = copy(node)

	for math in node.find_all("math"):
		alttext = math.get("alttext")
		if alttext:
			math.replace_with(f" {alttext} ")

	for noisy in node.find_all(
		["annotation", "annotation-xml", "semantics", "script", "style"]
	):
		noisy.decompose()

	return normalize_text(node.get_text(" ", strip=True))


def split_sentences(text: str) -> list[str]:
	text = normalize_text(text)
	if not text:
		return []

	text = protect_sentence_abbreviations(text)

	sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
	sentences = [
		restore_sentence_abbreviations(sentence).strip()
		for sentence in sentences
	]
	return [
		sentence
		for sentence in sentences
		if re.search(r"[.!?][\"')\]]*$", sentence)
	]


def trailing_lead_in(text: str) -> str:
	text = normalize_text(text)
	if not text:
		return ""

	protected_text = protect_sentence_abbreviations(text)
	match = list(re.finditer(r"[.!?][\"')\]]*(?=\s|$)", protected_text))
	if match:
		lead_in = protected_text[match[-1].end() :]
	else:
		lead_in = protected_text

	return normalize_text(restore_sentence_abbreviations(lead_in))


def nearby_sentences(anchor, direction: str) -> list[str]:
	sentences = []
	seen_texts = set()
	lead_in = ""
	checked_nearest_before_paragraph = False

	if direction == "before":
		candidates = anchor.find_all_previous(["p", "table"])
	else:
		candidates = anchor.find_all_next(["p", "table"])

	for candidate in candidates:
		if candidate is anchor:
			continue
		if is_context_boundary(candidate):
			break
		if candidate.name != "p" or not classes_contain(candidate, "ltx_p"):
			continue
		if candidate.find_parent(["table", "figure"]):
			continue

		text = clean_text_node(candidate)
		if not text or text in seen_texts:
			continue
		seen_texts.add(text)

		if direction == "before" and not checked_nearest_before_paragraph:
			lead_in = trailing_lead_in(text)
			checked_nearest_before_paragraph = True

		candidate_sentences = split_sentences(text)
		if direction == "before":
			candidate_sentences.reverse()

		for sentence in candidate_sentences:
			sentences.append(sentence)
			if len(sentences) >= CONTEXT_SENTENCE_LIMIT:
				break

		if len(sentences) >= CONTEXT_SENTENCE_LIMIT:
			break

	if direction == "before":
		sentences.reverse()
		if lead_in:
			sentences.append(lead_in)

	return sentences


def empty_surrounding_text() -> dict:
	return {
		"before": "",
		"after": "",
		"window": EQUATION_MARKER,
	}


def extract_surrounding_text(
	html_soup: BeautifulSoup | None,
	annotation_ids: list[str],
) -> tuple[dict, dict]:
	if html_soup is None:
		return empty_surrounding_text(), {
			"method": "Could not extract surrounding text because source HTML was not found.",
			"annotation_ids": annotation_ids,
			"anchor_id": None,
			"window": "0 paragraphs before / 0 paragraphs after",
		}

	if not annotation_ids:
		return empty_surrounding_text(), {
			"method": "Could not extract surrounding text because no annotation IDs were available.",
			"annotation_ids": annotation_ids,
			"anchor_id": None,
			"window": "0 paragraphs before / 0 paragraphs after",
		}

	anchor, matched_annotation_id = find_equation_anchor(html_soup, annotation_ids)
	if anchor is None:
		return empty_surrounding_text(), {
			"method": "Could not extract surrounding text because annotation IDs were not found in source HTML.",
			"annotation_ids": annotation_ids,
			"anchor_id": None,
			"window": "0 paragraphs before / 0 paragraphs after",
		}

	before = " ".join(nearby_sentences(anchor, "before"))
	after = " ".join(nearby_sentences(anchor, "after"))
	context = {
		"before": before,
		"after": after,
		"window": normalize_text(f"{before} {EQUATION_MARKER} {after}"),
	}
	audit = {
		"method": "Located equation by annotation id in source HTML and collected neighboring ltx_p sentences.",
		"annotation_ids": annotation_ids,
		"matched_annotation_id": matched_annotation_id,
		"anchor_id": anchor.get("id"),
		"window": f"up to {CONTEXT_SENTENCE_LIMIT} full sentences before / up to {CONTEXT_SENTENCE_LIMIT} full sentences after, stopping at neighboring equations",
	}

	return context, audit


def load_html_soup(paper_id: str) -> BeautifulSoup | None:
	html_file = HTML_DIR / f"{paper_id}.html"
	if not html_file.exists():
		return None
	return BeautifulSoup(
		html_file.read_text(encoding="utf-8", errors="ignore"),
		"html.parser",
	)


def make_entry(
	equation_key: str,
	equation: str,
	found_latex: bool,
	surrounding_text: dict,
	context_audit: dict,
) -> dict:
	audit_trail = [
		f"method extract_equations() found equation ({equation_key}) in 2_annotations.json.",
		{"context_extraction": context_audit},
	]

	return {
		"equation": equation,
		"meaning": "",
		"surrounding_text": surrounding_text,
		"symbols": {},
		"relations": {},
		"audit-trail": audit_trail,
	}


def extract_equations(
	input_file: Path,
	output_file: Path,
	paper_list_file: Path,
	target_min_equations: int,
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
		html_soup = load_html_soup(paper_id) if annotations_by_number else None

		for key, annotation_html_list in annotations_by_number.items():
			equation = format_equation(annotation_html_list)
			annotation_ids = extract_annotation_ids(annotation_html_list)
			surrounding_text, context_audit = extract_surrounding_text(
				html_soup,
				annotation_ids,
			)
			entry = make_entry(
				key,
				equation,
				bool(annotation_html_list),
				surrounding_text,
				context_audit,
			)
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


def extract_eqn_main(min_equations: int = TARGET_MIN_EQUATIONS) -> int:
	equation_count, paper_count, empty_paper_count = extract_equations(
		ANNOTATIONS_FILE,
		EQUATIONS_FILE,
		LIST_FILE,
		target_min_equations=min_equations,
	)
	print(f"Wrote {equation_count} equation entries to {EQUATIONS_FILE}")
	print(f"Output has {paper_count} papers and {equation_count} equations.")
	print(f"Empty/not found papers: {empty_paper_count}")

	return equation_count


# if __name__ == "__main__":
# 	extract_eqn_main()
