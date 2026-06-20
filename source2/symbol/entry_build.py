from symbol_config import CHUNKS_DIR, EQUATIONS_FILE, SYMBOLS_DIR
from symbol_io import build_symbol_files


def main() -> None:
	papers, symbols, enriched_chunks = build_symbol_files(
		EQUATIONS_FILE,
		SYMBOLS_DIR,
		CHUNKS_DIR,
	)
	print(f"Wrote {symbols} equation symbols for {papers} papers to {SYMBOLS_DIR}")
	print(f"Added symbols to {enriched_chunks} chunks in {CHUNKS_DIR}")


if __name__ == "__main__":
	main()
