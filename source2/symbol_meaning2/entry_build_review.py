from pathlib import Path
import sys


if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source2.symbol_meaning2.review import build_balanced_review_sample, write_review_sample
from source2.symbol_meaning2.symbol_config import REVIEW_SAMPLE_FILE, SYMBOL_MEANINGS_DIR


def main() -> None:
	rows = build_balanced_review_sample(SYMBOL_MEANINGS_DIR)
	write_review_sample(rows, REVIEW_SAMPLE_FILE)
	print(f"Wrote {len(rows)} review examples to {REVIEW_SAMPLE_FILE}")


if __name__ == "__main__":
	main()
