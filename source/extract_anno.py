from pathlib import Path
import json

from bs4 import BeautifulSoup


SPAN_SELECTOR = "span.ltx_tag_equation, span.ltx_tag_equationgroup"
TEX_SELECTOR = 'annotation[encoding="application/x-tex"]'
MAX_ENUMERATED_SPANS_PER_PAPER = 7


def equation_key(tag_text: str) -> str:
	tag_text = tag_text.strip()
	if tag_text.startswith("(") and tag_text.endswith(")"):
		return tag_text[1:-1]
	return tag_text


def extract_annotation_tags_by_number(html_content: str) -> dict[str, list[str]]:
	soup = BeautifulSoup(html_content, "html.parser")
	annotations_by_number = {}
	seen_blocks = set()
	span_count = 0

	for span in soup.select(SPAN_SELECTOR):
		if span_count >= MAX_ENUMERATED_SPANS_PER_PAPER:
			break

		tbody = span.find_parent("tbody")
		if not tbody:
			continue

		block_html = str(tbody)
		if block_html in seen_blocks:
			continue
		seen_blocks.add(block_html)

		key = equation_key(span.get_text(strip=True))
		annotations = [str(annotation) for annotation in tbody.select(TEX_SELECTOR)]
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


def main() -> None:
	project_dir = Path(__file__).resolve().parent.parent
	papers_dir = project_dir / "data" / "html"
	output_file = project_dir / "data" / "annotations.json"

	count = extract_annotations(papers_dir, output_file)
	print(f"Wrote {count} numbered annotation groups to {output_file}")


if __name__ == "__main__":
	main()
