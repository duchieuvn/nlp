from common import finish, read_records
from config import PREPROCESS_STEP_5_FILE, PREPROCESS_STEP_6_FILE
from s4b import MAX_MODEL_TOKENS, MODEL_NAME


def load_tokenizer():
	try:
		from transformers import AutoTokenizer
	except ImportError as error:
		raise RuntimeError(
			"Transformers is required for step 6. Run this script in the project ML environment."
		) from error
	return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def run() -> None:
	tokenizer = load_tokenizer()
	output = []
	for index, record in enumerate(read_records(PREPROCESS_STEP_5_FILE)):
		texts = record.get("model_texts")
		roles = record.get("text_roles")
		if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
			raise ValueError(f"Record {index} has invalid model_texts")
		if not isinstance(roles, list) or len(roles) != len(texts):
			raise ValueError(f"Record {index} has invalid text_roles")
		if texts:
			encoded = tokenizer(
				texts,
				padding=True,
				truncation=True,
				max_length=MAX_MODEL_TOKENS,
			)
			input_ids = encoded["input_ids"]
			attention_mask = encoded["attention_mask"]
		else:
			input_ids = []
			attention_mask = []
		output.append({
			"paper_id": record["paper_id"],
			"equation_id": record["equation_id"],
			"equation": record["equation"],
			"candidates": record["candidates"],
			"text_roles": roles,
			"input_ids": input_ids,
			"attention_mask": attention_mask,
		})
	finish(6, output, PREPROCESS_STEP_6_FILE)


if __name__ == "__main__":
	run()
