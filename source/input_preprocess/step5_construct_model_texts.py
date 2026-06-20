from common import finish, read_records, require_string
from config import PREPROCESS_STEP_4_FILE, PREPROCESS_STEP_5_FILE
from s4b import MAX_EQUATION_CHARACTERS, normalize_text


def run() -> None:
	output = []
	for index, record in enumerate(read_records(PREPROCESS_STEP_4_FILE)):
		candidates = record.get("candidates")
		if not isinstance(candidates, list):
			raise ValueError(f"Record {index} has invalid candidates")
		model_texts = []
		text_roles = []
		if candidates:
			model_texts.append(
				f"scientific context: {require_string(record, 'model_context', index)}"
			)
			text_roles.append("context")
			equation = normalize_text(require_string(record, "equation", index))[
				:MAX_EQUATION_CHARACTERS
			]
			if equation:
				model_texts.append(f"mathematical equation: {equation}")
				text_roles.append("equation")
			for candidate_index, candidate in enumerate(candidates):
				text = candidate.get("text") if isinstance(candidate, dict) else None
				if not isinstance(text, str):
					raise ValueError(f"Record {index} has an invalid candidate")
				model_texts.append(f"equation name: {text}")
				text_roles.append(f"candidate:{candidate_index}")
		output.append({
			**record,
			"model_texts": model_texts,
			"text_roles": text_roles,
		})
	finish(5, output, PREPROCESS_STEP_5_FILE)


if __name__ == "__main__":
	run()
