import hashlib

from .patterns import extract_regex_definition, mentions_alias
from .symbol_models import RELATION_LABELS


def bootstrap_regex_example(symbol: dict, sentence: str, paper_id: str) -> dict | None:
	definition, _, _ = extract_regex_definition(sentence, symbol.get("aliases", []))
	if not definition:
		return None
	return {
		"paper_id": paper_id,
		"canonical": symbol.get("canonical", ""),
		"phrase": definition,
		"sentence": sentence,
		"label": "DEFINES_COMPLETE_SYMBOL",
		"weak_label_source": "high_confidence_regex",
	}


def generate_hard_negatives(
	target_symbol: dict,
	other_symbols: list[dict],
	phrase: str,
	sentence: str,
	paper_id: str,
) -> list[dict]:
	return [{
		"paper_id": paper_id,
		"canonical": symbol.get("canonical", ""),
		"phrase": phrase,
		"sentence": sentence,
		"label": "NO_RELATION",
		"weak_label_source": "wrong_symbol_same_context",
		"hard_negative_for": target_symbol.get("canonical", ""),
	} for symbol in other_symbols if symbol.get("canonical") != target_symbol.get("canonical")]


def split_by_paper(examples: list[dict]) -> dict[str, list[dict]]:
	papers = sorted(
		{str(example["paper_id"]) for example in examples},
		key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
	)
	train_end = max(1, round(len(papers) * .8))
	validation_end = min(len(papers) - 1, train_end + max(1, round(len(papers) * .1)))
	assignments = {
		paper: "train" if index < train_end else "validation" if index < validation_end else "test"
		for index, paper in enumerate(papers)
	}
	output = {"train": [], "validation": [], "test": []}
	for example in examples:
		output[assignments[str(example["paper_id"])]].append(example)
	return output


def validate_relation_label(label: str) -> None:
	if label not in RELATION_LABELS:
		raise ValueError(f"Unknown relation label: {label}")
