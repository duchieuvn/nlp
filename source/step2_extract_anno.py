from pathlib import Path
import json

from bs4 import BeautifulSoup
from config import ANNOTATIONS_FILE, HTML_DIR


SPAN_SELECTOR = "span.ltx_tag_equation, span.ltx_tag_equationgroup"
TEX_SELECTOR = 'annotation[encoding="application/x-tex"]'
MAX_ENUMERATED_SPANS_PER_PAPER = 7


def equation_key(tag_text: str) -> str:
	tag_text = tag_text.strip()
	if tag_text.startswith("(") and tag_text.endswith(")"):
		return tag_text[1:-1]
	return tag_text


def annotation_tags(element) -> list[str]:
	return [str(annotation) for annotation in element.select(TEX_SELECTOR)]


def row_identity(row, fallback: int) -> tuple[str, str | int]:
	row_id = row.get("id")
	if row_id:
		return ("id", row_id)
	return ("index", fallback)


def split_tbody_annotations(tbody) -> list[tuple[str, list[str], tuple[str, str | int]]]:
	entries = []
	pending_annotations = []

	for row_index, row in enumerate(tbody.select("tr")):
		row_annotations = annotation_tags(row)
		span = row.select_one(SPAN_SELECTOR)

		if span:
			key = equation_key(span.get_text(strip=True))
			annotations = pending_annotations + row_annotations
			pending_annotations = []
			if key and annotations:
				entries.append((key, annotations, row_identity(row, row_index)))
			continue

		if row_annotations:
			pending_annotations.extend(row_annotations)

	return entries


def extract_annotation_tags_by_number(html_content: str) -> dict[str, list[str]]:
	soup = BeautifulSoup(html_content, "html.parser")
	annotations_by_number = {}
	seen_blocks = set()
	seen_multi_span_rows = set()
	span_count = 0

	for span in soup.select(SPAN_SELECTOR):
		if span_count >= MAX_ENUMERATED_SPANS_PER_PAPER:
			break

		tbody = span.find_parent("tbody")
		if not tbody:
			continue

		tbody_spans = tbody.select(SPAN_SELECTOR)
		if len(tbody_spans) > 1:
			for key, annotations, row_key in split_tbody_annotations(tbody):
				if span_count >= MAX_ENUMERATED_SPANS_PER_PAPER:
					break

				span_row_key = (id(tbody), row_key)
				if span_row_key in seen_multi_span_rows:
					continue
				seen_multi_span_rows.add(span_row_key)

				annotations_by_number.setdefault(key, []).extend(annotations)
				span_count += 1
			continue

		block_html = str(tbody)
		if block_html in seen_blocks:
			continue
		seen_blocks.add(block_html)

		key = equation_key(span.get_text(strip=True))
		annotations = annotation_tags(tbody)
		if not key or not annotations:
			continue

		annotations_by_number.setdefault(key, []).extend(annotations)
		span_count += 1

	return annotations_by_number


def extract_annotations(papers_dir: Path, output_file: Path) -> int:
	annotations_by_paper = {}

	for html_file in sorted(papers_dir.glob("*.html")):
		print(f"Processing paper: {html_file.name}")
		paper_id = html_file.stem
		html_content = html_file.read_text(encoding="utf-8", errors="ignore")
		annotations = extract_annotation_tags_by_number(html_content)

		if annotations:
			annotations_by_paper[paper_id] = annotations

	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(
		json.dumps(annotations_by_paper, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	return sum(len(equations) for equations in annotations_by_paper.values())


def extract_anno_main() -> int:
	count = extract_annotations(HTML_DIR, ANNOTATIONS_FILE)
	print(f"Wrote {count} numbered annotation groups to {ANNOTATIONS_FILE}")
	return count


# if __name__ == "__main__":
# 	extract_anno_main()
