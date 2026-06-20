import argparse
import json
from pathlib import Path

from batch_search import search_all_papers
from retrieval_config import (
	CHUNKS_DIR,
	RETRIEVAL_RESULTS_DIR,
	SYMBOLS_DIR,
)
from retrieval_models import SearchQuery
from retrieval_service import RetrievalService


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Search structured paper chunks")
	parser.add_argument("paper_id", nargs="?")
	parser.add_argument("query", nargs="?")
	parser.add_argument("--method", choices=("bm25", "tfidf"), default="bm25")
	parser.add_argument("--top-k", type=int, default=10)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=RETRIEVAL_RESULTS_DIR,
		help="Batch output directory (default: %(default)s)",
	)
	parser.add_argument("--section-id", action="append", dest="section_ids")
	parser.add_argument("--chunk-type", action="append", dest="chunk_types")
	parser.add_argument("--equation-id", action="append", dest="equation_ids")
	parser.add_argument("--symbol", action="append", dest="symbols")
	return parser


def main() -> None:
	arguments = _parser().parse_args()
	service = RetrievalService.from_directory(CHUNKS_DIR)
	if arguments.paper_id is None and arguments.query is None:
		papers, queries, results = search_all_papers(
			service,
			SYMBOLS_DIR,
			arguments.output_dir,
			method=arguments.method,
			top_k=arguments.top_k,
		)
		print(
			f"Wrote {results} results for {queries} equation queries "
			f"across {papers} papers to {arguments.output_dir}"
		)
		return
	if arguments.paper_id is None or arguments.query is None:
		raise SystemExit("paper_id and query must be provided together")
	query = SearchQuery(
		text=arguments.query,
		paper_id=arguments.paper_id,
		section_ids=arguments.section_ids,
		chunk_types=arguments.chunk_types,
		equation_ids=arguments.equation_ids,
		symbols=arguments.symbols,
		top_k=arguments.top_k,
	)
	print(json.dumps(
		[result.to_dict() for result in service.search(query, arguments.method)],
		ensure_ascii=False,
		indent=2,
	))


if __name__ == "__main__":
	main()
