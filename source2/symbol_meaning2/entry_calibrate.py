from pathlib import Path
import sys


if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source2.symbol_meaning2.review import calibrate_reviewed_file
from source2.symbol_meaning2.symbol_config import CHECKPOINT_DIR, REVIEW_SAMPLE_FILE


def main() -> None:
	calibration = calibrate_reviewed_file(REVIEW_SAMPLE_FILE, CHECKPOINT_DIR)
	print(
		f"Saved reviewed threshold {calibration['relation_threshold']:.6f} "
		f"at precision {calibration['accepted_precision']:.4f}"
	)


if __name__ == "__main__":
	main()
