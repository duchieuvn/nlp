from common import finish, read_records, require_string
from config import EQUATION_WINDOWS_FILE, PREPROCESS_STEP_1_FILE


def run() -> None:
	records = read_records(EQUATION_WINDOWS_FILE)
	for index, record in enumerate(records):
		if list(record) != ["paper_id", "equation_id", "equation", "window"]:
			raise ValueError(
				f"Record {index} must contain paper_id, equation_id, equation, and window "
				"in that order"
			)
		for key in ("paper_id", "equation_id", "equation", "window"):
			require_string(record, key, index)
	finish(1, records, PREPROCESS_STEP_1_FILE)


if __name__ == "__main__":
	run()
