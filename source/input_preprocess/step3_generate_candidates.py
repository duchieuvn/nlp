from common import finish, read_records, require_string
from config import PREPROCESS_STEP_2_FILE, PREPROCESS_STEP_3_FILE
import s4b


def run() -> None:
	output = []
	for index, record in enumerate(read_records(PREPROCESS_STEP_2_FILE)):
		before = require_string(record, "before_equation", index)
		candidates = s4b.candidate_phrases(before + s4b.EQUATION_MARKER)
		output.append({
			**record,
			"candidates": [
				{
					"text": candidate.text,
					"start": candidate.start,
					"end": candidate.end,
					"distance": candidate.distance,
					"cue_bonus": candidate.cue_bonus,
				}
				for candidate in candidates
			],
		})
	finish(3, output, PREPROCESS_STEP_3_FILE)


if __name__ == "__main__":
	run()
