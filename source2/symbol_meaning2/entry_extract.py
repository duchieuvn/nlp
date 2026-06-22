from pathlib import Path
import sys


if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source2.symbol_meaning2.cross_encoder import MathBERTCrossEncoder
from source2.symbol_meaning2.nlp import load_spacy_pipeline
from source2.symbol_meaning2.symbol_config import (
	CHECKPOINT_DIR,
	CHUNKS_DIR,
	SYMBOL_MEANINGS_DIR,
	SYMBOLS_DIR,
	TOP_K,
)
from source2.symbol_meaning2.symbol_io import extract_all_symbol_meanings
from source2.symbol_meaning2.symbol_retrieval import load_retrieval_service


def main() -> None:
	service = load_retrieval_service(CHUNKS_DIR)
	classifier = MathBERTCrossEncoder(CHECKPOINT_DIR)
	nlp = load_spacy_pipeline()
	print(
		f"Cross-encoder calibration: threshold="
		f"{classifier.calibration['threshold']:.6f}, "
		f"margin={classifier.calibration['margin']:.2f}, "
		f"scope={classifier.calibration['scope']}, "
		f"neural_acceptance_enabled="
		f"{classifier.calibration['acceptance_enabled']}"
	)
	papers, symbols, definitions = extract_all_symbol_meanings(
		service, SYMBOLS_DIR, SYMBOL_MEANINGS_DIR, top_k=TOP_K,
		nlp=nlp, classifier=classifier,
	)
	print(
		f"Wrote {definitions} definitions for {symbols} symbols across "
		f"{papers} papers to {SYMBOL_MEANINGS_DIR}"
	)


if __name__ == "__main__":
	main()
