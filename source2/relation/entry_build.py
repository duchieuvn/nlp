from relation_config import RELATIONS_DIR, STRUCTURED_PAPERS_DIR, SYMBOLS_DIR
from relation_io import build_all_relation_files


def main() -> None:
	papers, pairs, strong = build_all_relation_files(
		STRUCTURED_PAPERS_DIR,
		SYMBOLS_DIR,
		RELATIONS_DIR,
	)
	print(
		f"Wrote {pairs} directed equation relations across {papers} papers "
		f"({strong} strong) to {RELATIONS_DIR}"
	)


if __name__ == "__main__":
	main()
