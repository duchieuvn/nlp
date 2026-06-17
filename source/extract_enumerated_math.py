from pathlib import Path
import json
from bs4 import BeautifulSoup
from config import HTML_DIR, DATA_DIR


def extract_enumerated_spans(papers_dir: Path, output_file: Path) -> int:
	spans_by_paper = {}

	for html_file in sorted(papers_dir.glob("*.html")):
		print(f"Processing paper: {html_file.name}")
		paper_id = html_file.stem
		with html_file.open("r", encoding="utf-8", errors="ignore") as handle:
			html_content = handle.read()

		soup = BeautifulSoup(html_content, "html.parser")


		# Find all span tags with either class using a CSS selector
		spans = soup.select("span.ltx_tag_equation, span.ltx_tag_equationgroup")

		tbodies = []
		seen = set()

		for span in spans:
			tbody = span.find_parent("tbody")
			if tbody:
				tbody_html = str(tbody)
				if tbody_html not in seen:
					seen.add(tbody_html)
					tbodies.append(tbody_html)

		if tbodies:
			spans_by_paper[paper_id] = tbodies

	output_file.write_text(json.dumps(spans_by_paper, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	return sum(len(tbody_list) for tbody_list in spans_by_paper.values())


def main() -> None:
	papers_dir = Path(HTML_DIR)
	output_file = Path(DATA_DIR) / "html_math.json"

	count = extract_enumerated_spans(papers_dir, output_file)
	print(f"Wrote {count} matching <tbody> blocks to {output_file}")


if __name__ == "__main__":
	main()
