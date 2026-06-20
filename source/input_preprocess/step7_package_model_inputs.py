from common import finish, read_records
from config import PREPROCESS_STEP_6_FILE, PREPROCESS_STEP_7_FILE
from s4b import MODEL_NAME


def run() -> None:
	output = []
	for index, record in enumerate(read_records(PREPROCESS_STEP_6_FILE)):
		input_ids = record.get("input_ids")
		attention_mask = record.get("attention_mask")
		roles = record.get("text_roles")
		if not isinstance(input_ids, list) or not isinstance(attention_mask, list):
			raise ValueError(f"Record {index} has invalid encoded arrays")
		if not isinstance(roles, list):
			raise ValueError(f"Record {index} has invalid text_roles")
		if len(input_ids) != len(attention_mask) or len(input_ids) != len(roles):
			raise ValueError(f"Record {index} encoded batch sizes do not match")
		for row, mask in zip(input_ids, attention_mask):
			if not isinstance(row, list) or not isinstance(mask, list) or len(row) != len(mask):
				raise ValueError(f"Record {index} has mismatched token rows")
		output.append({
			"paper_id": record["paper_id"],
			"equation_id": record["equation_id"],
			"equation": record["equation"],
			"candidates": record["candidates"],
			"model_name": MODEL_NAME,
			"text_roles": roles,
			"model_inputs": {
				"input_ids": input_ids,
				"attention_mask": attention_mask,
			},
		})
	finish(7, output, PREPROCESS_STEP_7_FILE)


if __name__ == "__main__":
	run()
