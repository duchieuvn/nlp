from common import finish, read_records, require_string
from config import PREPROCESS_STEP_1_FILE, PREPROCESS_STEP_2_FILE


START_TOKEN = "<starteqn>"
END_TOKEN = "<endeqn>"


def split_window(window: str, record_index: int) -> tuple[str, str, str]:
	if window.count(START_TOKEN) != 1 or window.count(END_TOKEN) != 1:
		raise ValueError(f"Record {record_index} must contain one equation token pair")
	before, remainder = window.split(START_TOKEN, 1)
	equation, after = remainder.split(END_TOKEN, 1)
	return before, equation.strip(), after


def run() -> None:
	output = []
	for index, record in enumerate(read_records(PREPROCESS_STEP_1_FILE)):
		window = require_string(record, "window", index)
		before, embedded_equation, after = split_window(window, index)
		if embedded_equation != require_string(record, "equation", index).strip():
			raise ValueError(f"Record {index} embedded equation does not match equation field")
		output.append({
			**record,
			"before_equation": before,
			"embedded_equation": embedded_equation,
			"after_equation": after,
		})
	finish(2, output, PREPROCESS_STEP_2_FILE)


if __name__ == "__main__":
	run()
