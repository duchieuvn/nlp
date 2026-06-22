from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from .config import (
	LABELS,
	MAX_REJECTED_CANDIDATES_PER_SYMBOL,
	RANDOM_SEED,
	TRAIN_FRACTION,
	VALIDATION_FRACTION,
)


@dataclass(frozen=True)
class RelationExample:
	paper_id: str
	equation_id: str
	canonical: str
	original_latex: str
	equation: str
	phrase: str
	sentence: str
	label: str
	weak_label_source: str
	has_modifiers: bool

	@property
	def symbol_context(self) -> str:
		return " | ".join((
			f"original LaTeX: {self.original_latex}",
			f"canonical: {self.canonical}",
			f"equation: {self.equation}",
			f"context: {self.sentence}",
		))

	@property
	def candidate_context(self) -> str:
		return f"candidate phrase: {self.phrase} | evidence: {self.sentence}"

	def to_dict(self) -> dict:
		return asdict(self)


def _symbol_registry(symbols_dir: Path) -> dict[tuple[str, str, str], tuple[dict, str]]:
	registry = {}
	for path in sorted(symbols_dir.glob("*.json")):
		payload = json.loads(path.read_text(encoding="utf-8"))
		for equation in payload["equations"]:
			for symbol in equation["symbols"]:
				key = (payload["paper_id"], equation["equation_id"], symbol["canonical"])
				registry[key] = symbol, equation["latex"]
	return registry


def _has_modifiers(symbol: dict) -> bool:
	canonical = symbol.get("canonical", "")
	latex = " ".join(symbol.get("latex_forms", []))
	return bool(
		"_" in canonical
		or re.search(r"[_^]", latex)
		or re.search(r"\\(?:bar|vec|hat|tilde|dot|ddot|overline)\b", latex)
	)


def _base_aliases(symbol: dict) -> list[str]:
	aliases = []
	for latex in symbol.get("latex_forms", []):
		match = re.match(
			r"(?:\\(?:bar|vec|hat|tilde|dot|ddot|overline)\s*\{)?"
			r"(?P<base>\\[A-Za-z]+|[A-Za-z])",
			latex.strip(),
		)
		if match:
			aliases.append(match.group("base"))
	canonical = symbol.get("canonical", "")
	if canonical:
		aliases.append(canonical.split("_")[0])
	return list(dict.fromkeys(filter(None, aliases)))


def _exact_symbol_present(symbol: dict, text: str) -> bool:
	return any(form.strip() and form.strip() in text for form in symbol.get("latex_forms", []))


def _alias_present(symbol: dict, text: str) -> bool:
	for alias in symbol.get("aliases", []):
		alias = alias.strip()
		if not alias:
			continue
		if alias.startswith("\\") or len(alias) > 1:
			if alias in text:
				return True
		elif re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text):
			return True
	return False


def _extract_phrase(text: str) -> str:
	patterns = (
		r"\b(?:denotes?|represents?|means?|refers?\s+to|stands?\s+for|is\s+defined\s+as|are\s+defined\s+as|let\s+\S+\s+be)\s+(?P<phrase>[^.;]{1,180})",
		r"\b(?:is|are)\s+(?:the|a|an)\s+(?P<phrase>[^.;]{1,180})",
		r"\b(?:denote|define|represent|write)\s+(?P<phrase>[^.;]{1,180}?)\s+(?:as|by)\s+",
	)
	for pattern in patterns:
		match = re.search(pattern, text, re.IGNORECASE)
		if match:
			phrase = re.sub(r"\s+", " ", match.group("phrase")).strip(" ,;:.()")
			phrase = re.split(
				r",?\s+and\s+(?:\\?[A-Za-z][^,]{0,40})\s+"
				r"(?:is|are|denotes?|represents?)\b",
				phrase,
				maxsplit=1,
				flags=re.IGNORECASE,
			)[0]
			return phrase[:500]
	return re.sub(r"\s+", " ", text).strip()[:500]


def _modifier_phrase(symbol: dict, text: str) -> tuple[str, str] | None:
	canonical = symbol.get("canonical", "").casefold()
	if "_sup_l" in canonical or re.search(r"\^\{?L\}?", " ".join(symbol.get("latex_forms", []))):
		if re.search(r"\bleft\b", text, re.IGNORECASE):
			return "left", "QUALIFIES_SUPERSCRIPT"
	if "_sup_r" in canonical or re.search(r"\^\{?R\}?", " ".join(symbol.get("latex_forms", []))):
		if re.search(r"\bright\b", text, re.IGNORECASE):
			return "right", "QUALIFIES_SUPERSCRIPT"
	if re.search(r"\b(?:subscript|indexed by|index)\b", text, re.IGNORECASE):
		match = re.search(r"\b(?:subscript|indexed by|index)\b[^.;]{0,80}", text, re.IGNORECASE)
		return (match.group(0), "QUALIFIES_SUBSCRIPT") if match else None
	return None


def _looks_like_base_definition(symbol: dict, text: str) -> bool:
	if not _has_modifiers(symbol) or _exact_symbol_present(symbol, text):
		return False
	for alias in _base_aliases(symbol):
		pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9_^])\s+(?:denotes?|represents?|is\s+(?:the|a|an)\b)"
		if re.search(pattern, text, re.IGNORECASE):
			return True
	return False


def _add_example(
	output: list[RelationExample],
	seen: set[tuple],
	registry: dict,
	paper_id: str,
	equation_id: str,
	canonical: str,
	phrase: str,
	sentence: str,
	label: str,
	source: str,
) -> None:
	item = registry.get((paper_id, equation_id, canonical))
	if item is None or label not in LABELS:
		return
	phrase = re.sub(r"\s+", " ", phrase).strip()
	sentence = re.sub(r"\s+", " ", sentence).strip()
	if not phrase or not sentence:
		return
	key = (paper_id, equation_id, canonical, phrase.casefold(), sentence.casefold(), label)
	if key in seen:
		return
	seen.add(key)
	symbol, equation = item
	output.append(RelationExample(
		paper_id, equation_id, canonical,
		(symbol.get("latex_forms") or [canonical])[0], equation,
		phrase, sentence, label, source, _has_modifiers(symbol),
	))


def build_examples(
	symbols_dir: Path,
	meanings_dir: Path,
	rejected_file: Path,
) -> list[RelationExample]:
	registry = _symbol_registry(symbols_dir)
	output: list[RelationExample] = []
	seen: set[tuple] = set()

	for path in sorted(meanings_dir.glob("*.json")):
		payload = json.loads(path.read_text(encoding="utf-8"))
		for equation in payload["equations"]:
			for record in equation["symbols"]:
				if record.get("definition"):
					_add_example(
						output, seen, registry, payload["paper_id"],
						equation["equation_id"], record["canonical"],
						record["definition"], record["source_text"],
						"DEFINES_COMPLETE_SYMBOL", "accepted_regex",
					)

	payload = json.loads(rejected_file.read_text(encoding="utf-8"))
	for record in payload["rejected_symbol_meanings"]:
		item = registry.get((record["paper_id"], record["equation_id"], record["canonical"]))
		if item is None:
			continue
		symbol, _ = item
		for candidate in record.get("candidates", [])[:MAX_REJECTED_CANDIDATES_PER_SYMBOL]:
			text = candidate.get("text", "").strip()
			if not text:
				continue
			modifier = _modifier_phrase(symbol, text)
			if modifier and _exact_symbol_present(symbol, text):
				phrase, label = modifier
			elif _looks_like_base_definition(symbol, text):
				phrase, label = _extract_phrase(text), "DEFINES_BASE"
			elif candidate.get("alias_mentioned") and _alias_present(symbol, text):
				phrase, label = _extract_phrase(text), "RELATED_NOT_DEFINITION"
			else:
				phrase, label = _extract_phrase(text), "NO_RELATION"
			_add_example(
				output, seen, registry, record["paper_id"], record["equation_id"],
				record["canonical"], phrase, text, label, "rejected_bm25",
			)
	return output


def split_by_paper(examples: list[RelationExample]) -> dict[str, list[RelationExample]]:
	papers = sorted(
		{example.paper_id for example in examples},
		key=lambda paper_id: hashlib.sha256(
			f"{RANDOM_SEED}:{paper_id}".encode()
		).hexdigest(),
	)
	train_end = max(1, round(len(papers) * TRAIN_FRACTION))
	validation_count = max(1, round(len(papers) * VALIDATION_FRACTION))
	validation_end = min(len(papers) - 1, train_end + validation_count)
	assignments = {
		paper_id: (
			"train" if index < train_end
			else "validation" if index < validation_end
			else "test"
		)
		for index, paper_id in enumerate(papers)
	}
	output = {"train": [], "validation": [], "test": []}
	for example in examples:
		output[assignments[example.paper_id]].append(example)
	return output


def dataset_summary(examples: list[RelationExample], splits: dict) -> dict:
	return {
		"weakly_supervised": True,
		"example_count": len(examples),
		"paper_count": len({example.paper_id for example in examples}),
		"label_counts": dict(Counter(example.label for example in examples)),
		"splits": {
			name: {
				"examples": len(rows),
				"papers": len({row.paper_id for row in rows}),
				"label_counts": dict(Counter(row.label for row in rows)),
			}
			for name, rows in splits.items()
		},
	}
