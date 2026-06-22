import argparse
from pathlib import Path

try:
	from .postprocessing_io import postprocess_directory, summarize_directory
except ImportError:
	from postprocessing_io import postprocess_directory, summarize_directory


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MEANINGS_DIR = PROJECT_DIR / "data" / "source2" / "equation_meanings"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "postprocessing" / "equation_meanings"


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Conservatively shorten extracted equation meanings",
	)
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_MEANINGS_DIR)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--paper-id", action="append", dest="paper_ids")
	return parser


def main() -> None:
	arguments = _parser().parse_args()
	papers, records, changed = postprocess_directory(
		arguments.input_dir,
		arguments.output_dir,
		set(arguments.paper_ids) if arguments.paper_ids else None,
	)
	print(
		f"Postprocessed {records} meanings across {papers} papers; "
		f"shortened {changed} meanings in {arguments.output_dir}"
	)
	summary = summarize_directory(
		arguments.output_dir,
		set(arguments.paper_ids) if arguments.paper_ids else None,
	)
	print(
		f"Phrase results: nonempty={summary['nonempty']}, empty={summary['empty']}, "
		f"shortened={summary['shortened']}, flagged={summary['flagged']}, "
		f"word_lengths={summary['phrase_words']}, "
		f"validation_failures={summary['validation_failures']}"
	)
	print(f"Strategies: {summary['strategies']}")


if __name__ == "__main__":
	main()
