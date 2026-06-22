import argparse
from pathlib import Path

try:
	from .exporter import export_final_data
except ImportError:
	from exporter import export_final_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Build the strict final dataset")
	parser.add_argument(
		"--equations-file", type=Path, default=DATA_DIR / "3_equations.json"
	)
	parser.add_argument(
		"--paper-list-file", type=Path, default=DATA_DIR / "paper_list_46.txt"
	)
	parser.add_argument(
		"--meanings-dir",
		type=Path,
		default=DATA_DIR / "postprocessing" / "equation_meanings",
	)
	parser.add_argument(
		"--symbols-dir",
		type=Path,
		default=DATA_DIR / "source2" / "symbol_meanings",
	)
	parser.add_argument(
		"--relations-dir",
		type=Path,
		default=DATA_DIR / "source2" / "relations",
	)
	parser.add_argument(
		"--output-file", type=Path, default=DATA_DIR / "final_data.json"
	)
	return parser


def main() -> None:
	arguments = _parser().parse_args()
	papers, equations, symbols, relations = export_final_data(
		arguments.equations_file,
		arguments.paper_list_file,
		arguments.meanings_dir,
		arguments.symbols_dir,
		arguments.relations_dir,
		arguments.output_file,
	)
	print(
		f"Wrote {equations} equations across {papers} papers with "
		f"{symbols} symbols and {relations} relations to "
		f"{arguments.output_file}"
	)


if __name__ == "__main__":
	main()
