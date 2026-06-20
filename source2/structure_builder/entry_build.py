from config import (
	ANNOTATIONS_FILE,
	EQUATIONS_FILE,
	HTML_DIR,
	STRUCTURED_PAPERS_DIR,
)
from document_builder import build_corpus, write_papers


def main() -> None:
	corpus = build_corpus(ANNOTATIONS_FILE, EQUATIONS_FILE, HTML_DIR)
	write_papers(corpus, STRUCTURED_PAPERS_DIR)
	report = corpus["build_report"]
	print(
		f"Wrote {report['built_paper_count']} papers and "
		f"{report['target_equation_count']} equations to {STRUCTURED_PAPERS_DIR}"
	)
	print(
		f"Resolved {report['resolved_equation_count']} equations; "
		f"unresolved: {report['unresolved_equation_count']}"
	)


if __name__ == "__main__":
	main()
