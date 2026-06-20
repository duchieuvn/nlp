from meaning_config import (
	EQUATION_MEANINGS_DIR,
	EQUATIONS_FILE,
	RETRIEVAL_RESULTS_DIR,
)
from meaning_io import extract_all_meanings
from mathbert_reranker import MathBERTReranker


def main() -> None:
	reranker = MathBERTReranker()
	print(f"Loaded {reranker.model_name} on {reranker.device}")
	papers, equations, meanings = extract_all_meanings(
		RETRIEVAL_RESULTS_DIR,
		EQUATIONS_FILE,
		EQUATION_MEANINGS_DIR,
		reranker=reranker,
	)
	print(
		f"Wrote {meanings} meanings for {equations} equations across "
		f"{papers} papers to {EQUATION_MEANINGS_DIR}"
	)


if __name__ == "__main__":
	main()
