from symbol_config import CHUNKS_DIR, SYMBOL_MEANINGS_DIR, SYMBOLS_DIR, TOP_K
from symbol_io import extract_all_symbol_meanings
from symbol_retrieval import load_retrieval_service
from spacy_fallback import load_spacy_pipeline


def main() -> None:
	service = load_retrieval_service(CHUNKS_DIR)
	nlp = load_spacy_pipeline()
	print("spaCy dependency fallback: " + ("enabled" if nlp else "unavailable"))
	papers, symbols, definitions = extract_all_symbol_meanings(
		service,
		SYMBOLS_DIR,
		SYMBOL_MEANINGS_DIR,
		top_k=TOP_K,
		nlp=nlp,
	)
	print(
		f"Wrote {definitions} definitions for {symbols} symbols across "
		f"{papers} papers to {SYMBOL_MEANINGS_DIR}"
	)


if __name__ == "__main__":
	main()
