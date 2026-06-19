from __future__ import annotations

from pathlib import Path
import json
import re
import sys
from typing import Any

from config import EQUATION_MEANINGS_B_FILE, EQUATIONS_FILE


QA_MODEL = "deepset/roberta-base-squad2"
QA_QUESTION = "What is the name of the equation or state?"
CONFIDENCE_THRESHOLD = 0.50
EQUATION_MARKER = "[EQUATION]"
DEFAULT_MAX_LENGTH = 512
LEADING_DROP_WORDS = {"a", "an", "our", "the", "their", "these", "this", "those"}


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def clean_candidate(text: str) -> str:
	text = normalize_text(text)
	text = re.sub(r"^[,;:.\s]+|[,;:.\s]+$", "", text)
	words = text.split()
	while words and words[0].lower() in LEADING_DROP_WORDS:
		words.pop(0)
	return " ".join(words)


def candidate_rejection_reason(text: str) -> str:
	words = re.findall(r"[A-Za-z]+", text)
	if not words:
		return "The answer contains no words"
	if len(words) > 12:
		return "The answer is longer than 12 words"
	if re.search(r"\b(?:eq|fig|sec|app|ref)s?\.?\s*\(?\s*\d+", text, re.I):
		return "The answer resembles a numbered cross-reference"
	if re.search(r"[()[\]{}]", text):
		return "The answer contains bracketed mathematical syntax"

	compact = re.sub(r"\s+", "", text)
	alpha_count = len(re.findall(r"[A-Za-z]", compact))
	symbol_count = len(compact) - alpha_count
	if alpha_count < 3 or (symbol_count > alpha_count and alpha_count < 8):
		return "The answer contains too little natural language"
	return ""


def _usable_model_max_length(tokenizer: Any) -> int:
	max_length = getattr(tokenizer, "model_max_length", DEFAULT_MAX_LENGTH)
	if not isinstance(max_length, int) or max_length <= 0 or max_length > 100_000:
		return DEFAULT_MAX_LENGTH
	return min(max_length, DEFAULT_MAX_LENGTH)


def model_input_window(
	window: str,
	tokenizer: Any,
	max_length: int,
	question: str = QA_QUESTION,
) -> str:
	"""Build a marker-centered QA context that fits the pair token limit."""
	window = normalize_text(window)
	if not window:
		return ""

	question_ids = tokenizer.encode(question, add_special_tokens=False)
	pair_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
	context_limit = max_length - len(question_ids) - pair_special_tokens
	if context_limit <= 0:
		raise RuntimeError(f"Invalid QA model maximum length: {max_length}")

	if EQUATION_MARKER not in window:
		token_ids = tokenizer.encode(window, add_special_tokens=False)[:context_limit]
		return tokenizer.decode(
			token_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		).strip()

	before, after = window.split(EQUATION_MARKER, 1)
	before_ids = tokenizer.encode(before, add_special_tokens=False)
	marker_ids = tokenizer.encode(EQUATION_MARKER, add_special_tokens=False)
	after_ids = tokenizer.encode(after, add_special_tokens=False)
	if len(marker_ids) > context_limit:
		raise RuntimeError("The QA tokenizer cannot fit the equation marker")

	available = context_limit - len(marker_ids)
	before_limit = min(len(before_ids), available // 2)
	after_limit = min(len(after_ids), available - before_limit)
	remaining = available - before_limit - after_limit
	before_limit += min(len(before_ids) - before_limit, remaining)
	remaining = available - before_limit - after_limit
	after_limit += min(len(after_ids) - after_limit, remaining)

	token_ids = before_ids[-before_limit:] if before_limit else []
	token_ids += marker_ids
	token_ids += after_ids[:after_limit]
	return tokenizer.decode(
		token_ids,
		skip_special_tokens=False,
		clean_up_tokenization_spaces=False,
	).strip()


def load_qa_pipeline(model: str = QA_MODEL) -> Any:
	try:
		import torch
		from transformers import pipeline
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"step4_b.py requires transformers and a CUDA-enabled torch install."
		) from exc

	if not torch.cuda.is_available():
		raise RuntimeError("step4_b.py requires an NVIDIA CUDA device for QA inference.")

	try:
		return pipeline(
			"question-answering",
			model=model,
			tokenizer=model,
			device=0,
		)
	except Exception as exc:
		raise RuntimeError(f"Could not load extractive QA model {model}: {exc}") from exc


def _blank_result(status: str, reason: str, **values: Any) -> dict[str, Any]:
	return {
		"meaning": "",
		"raw_answer": values.pop("raw_answer", ""),
		"confidence": values.pop("confidence", 0.0),
		"status": status,
		"reason": reason,
		**values,
	}


def predict_meaning(
	window: str,
	qa_pipeline: Any,
	threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
	max_length = _usable_model_max_length(qa_pipeline.tokenizer)
	model_window = model_input_window(
		window,
		qa_pipeline.tokenizer,
		max_length,
	)
	if not model_window:
		return _blank_result(
			"missing_context",
			"No surrounding_text.window was available",
		)

	try:
		prediction = qa_pipeline(question=QA_QUESTION, context=model_window)
	except Exception as exc:
		return _blank_result("error", f"QA inference failed: {exc}")

	raw_answer = str(prediction.get("answer", ""))
	try:
		confidence = float(prediction.get("score", 0.0))
		start = int(prediction.get("start", -1))
		end = int(prediction.get("end", -1))
	except (TypeError, ValueError):
		return _blank_result(
			"invalid_span",
			"The QA model returned malformed score or offsets",
			raw_answer=raw_answer,
		)

	span_is_valid = (
		0 <= start < end <= len(model_window)
		and model_window[start:end] == raw_answer
	)
	common = {
		"raw_answer": raw_answer,
		"confidence": confidence,
		"start": start,
		"end": end,
		"source_text": model_window,
	}
	if not span_is_valid:
		return _blank_result(
			"invalid_span",
			"The QA answer does not match its context offsets",
			**common,
		)
	if confidence < threshold:
		return _blank_result(
			"low_confidence",
			f"QA confidence is below the {threshold:.2f} threshold",
			**common,
		)

	meaning = clean_candidate(raw_answer)
	rejection_reason = candidate_rejection_reason(meaning)
	if rejection_reason:
		return _blank_result(
			"rejected_candidate",
			rejection_reason,
			**common,
		)

	return {
		"meaning": meaning,
		"raw_answer": raw_answer,
		"confidence": confidence,
		"status": "accepted",
		"start": start,
		"end": end,
		"source_text": model_window,
	}


def meaning_audit(
	result: dict[str, Any],
	model: str,
	threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
	audit = {
		"method": "Local extractive question answering on surrounding_text.window",
		"model": model,
		"question": QA_QUESTION,
		"candidate": result["meaning"],
		"raw_answer": result.get("raw_answer", ""),
		"confidence": result["confidence"],
		"threshold": threshold,
		"strategy": "extractive_qa",
		"status": result["status"],
	}
	if "start" in result:
		audit["start"] = result["start"]
		audit["end"] = result["end"]
	if result.get("source_text"):
		audit["source_text"] = result["source_text"]
	if result.get("reason"):
		audit["reason"] = result["reason"]
	return {"meaning_extraction": audit}


def extract_meanings(
	input_file: Path = EQUATIONS_FILE,
	output_file: Path = EQUATION_MEANINGS_B_FILE,
	model: str = QA_MODEL,
	threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[int, int, int]:
	qa_pipeline = load_qa_pipeline(model)
	data = json.loads(input_file.read_text(encoding="utf-8"))
	visited_count = 0
	filled_count = 0

	for paper_equations in data.values():
		for entry in paper_equations.values():
			visited_count += 1
			window = entry.get("surrounding_text", {}).get("window", "")
			result = predict_meaning(window, qa_pipeline, threshold)
			entry["meaning"] = result["meaning"]
			if result["meaning"]:
				filled_count += 1

			audit_trail = entry.setdefault("audit-trail", [])
			audit_trail.append(meaning_audit(result, model, threshold))

	blank_count = visited_count - filled_count
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(
		json.dumps(data, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	return visited_count, filled_count, blank_count


def extract_meaning_main() -> int:
	visited_count, filled_count, blank_count = extract_meanings()
	print(f"Visited equations: {visited_count}")
	print(f"Meanings filled: {filled_count}")
	print(f"Meanings left blank: {blank_count}")
	print(f"Wrote equation meanings to {EQUATION_MEANINGS_B_FILE}")
	return filled_count


if __name__ == "__main__":
	try:
		extract_meaning_main()
	except (OSError, RuntimeError, ValueError) as exc:
		print(exc, file=sys.stderr)
		sys.exit(1)
