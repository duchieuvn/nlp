from common import finish, read_records, require_string
from config import PREPROCESS_STEP_3_FILE, PREPROCESS_STEP_4_FILE
from export_equation_windows import END_EQUATION_TOKEN, START_EQUATION_TOKEN
from s4b import MAX_CONTEXT_CHARACTERS, normalize_text


BEFORE_CONTEXT_CHARACTERS = 1200
AFTER_CONTEXT_CHARACTERS = 600


def build_context(before: str, equation: str, after: str) -> str:
	token_overhead = len(START_EQUATION_TOKEN) + len(END_EQUATION_TOKEN) + 2
	equation_budget = max(0, MAX_CONTEXT_CHARACTERS - token_overhead)
	bounded_equation = equation[:equation_budget]
	embedded = f"{START_EQUATION_TOKEN} {bounded_equation} {END_EQUATION_TOKEN}"
	remaining = max(0, MAX_CONTEXT_CHARACTERS - len(embedded))
	before_budget = min(len(before), BEFORE_CONTEXT_CHARACTERS, remaining * 2 // 3)
	after_budget = min(len(after), AFTER_CONTEXT_CHARACTERS, remaining - before_budget)
	unused = remaining - before_budget - after_budget
	before_budget += min(
		unused,
		max(0, min(len(before), BEFORE_CONTEXT_CHARACTERS) - before_budget),
	)
	unused = remaining - before_budget - after_budget
	after_budget += min(
		unused,
		max(0, min(len(after), AFTER_CONTEXT_CHARACTERS) - after_budget),
	)
	return normalize_text(
		f"{before[-before_budget:] if before_budget else ''} "
		f"{embedded} {after[:after_budget]}"
	)


def run() -> None:
	output = []
	for index, record in enumerate(read_records(PREPROCESS_STEP_3_FILE)):
		context = build_context(
			require_string(record, "before_equation", index),
			require_string(record, "embedded_equation", index),
			require_string(record, "after_equation", index),
		)
		output.append({**record, "model_context": context})
	finish(4, output, PREPROCESS_STEP_4_FILE)


if __name__ == "__main__":
	run()
