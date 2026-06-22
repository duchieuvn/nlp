import json
from pathlib import Path
from typing import Any


VALID_GRADES = {"none", "potential", "strong"}
FINAL_FIELDS = ("equation", "meaning", "symbols", "relations")


def _equation_sort_key(equation_id: str) -> tuple[int, int | str]:
	return (0, int(equation_id)) if equation_id.isdigit() else (1, equation_id)


def _load_json(path: Path) -> Any:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f"Cannot load JSON from {path}: {error}") from error


def _load_paper_order(paper_list_file: Path, expected: set[str]) -> list[str]:
	try:
		lines = paper_list_file.read_text(encoding="utf-8").splitlines()
	except OSError as error:
		raise ValueError(f"Cannot load paper list from {paper_list_file}: {error}") from error
	ordered = []
	seen = set()
	for line in lines:
		paper_id = line.strip()
		if paper_id.startswith("arXiv:"):
			paper_id = paper_id.removeprefix("arXiv:").strip()
		if paper_id in expected and paper_id not in seen:
			ordered.append(paper_id)
			seen.add(paper_id)
	missing = expected - seen
	if missing:
		raise ValueError(f"Papers missing from paper list: {sorted(missing)}")
	return ordered


def _load_component_directory(directory: Path, component: str) -> dict[str, dict]:
	if not directory.is_dir():
		raise ValueError(f"Missing {component} directory: {directory}")
	payloads = {}
	for path in sorted(directory.glob("*.json")):
		payload = _load_json(path)
		if not isinstance(payload, dict):
			raise ValueError(f"Invalid {component} payload: {path}")
		paper_id = payload.get("paper_id")
		if paper_id != path.stem:
			raise ValueError(
				f"{component} paper ID {paper_id!r} does not match {path.name}"
			)
		if paper_id in payloads:
			raise ValueError(f"Duplicate {component} paper: {paper_id}")
		payloads[paper_id] = payload
	return payloads


def _index_equations(payload: dict, paper_id: str, component: str) -> dict[str, dict]:
	equations = payload.get("equations")
	if not isinstance(equations, list):
		raise ValueError(f"Invalid {component} equations for paper {paper_id}")
	indexed = {}
	for equation in equations:
		if not isinstance(equation, dict):
			raise ValueError(f"Invalid {component} equation for paper {paper_id}")
		equation_id = equation.get("equation_id")
		if not isinstance(equation_id, str) or not equation_id:
			raise ValueError(f"Invalid {component} equation ID for paper {paper_id}")
		if equation_id in indexed:
			raise ValueError(
				f"Duplicate {component} equation {paper_id}:{equation_id}"
			)
		indexed[equation_id] = equation
	return indexed


def _validate_component_coverage(
	base: dict[str, dict],
	payloads: dict[str, dict],
	component: str,
) -> dict[str, dict[str, dict]]:
	expected_papers = {paper_id for paper_id, equations in base.items() if equations}
	actual_papers = set(payloads)
	if actual_papers != expected_papers:
		missing = sorted(expected_papers - actual_papers)
		extra = sorted(actual_papers - expected_papers)
		raise ValueError(
			f"{component} paper mismatch: missing={missing}, extra={extra}"
		)
	indexed = {}
	for paper_id in sorted(expected_papers):
		equations = _index_equations(payloads[paper_id], paper_id, component)
		expected_ids = set(base[paper_id])
		if set(equations) != expected_ids:
			raise ValueError(
				f"{component} equation mismatch for {paper_id}: "
				f"expected={sorted(expected_ids)}, actual={sorted(equations)}"
			)
		indexed[paper_id] = equations
	return indexed


def _build_symbols(entry: dict, paper_id: str, equation_id: str) -> dict[str, str]:
	if not isinstance(entry.get("symbols"), list):
		raise ValueError(f"Invalid symbols for {paper_id}:{equation_id}")
	result = {}
	seen = set()
	for symbol in entry["symbols"]:
		if not isinstance(symbol, dict):
			raise ValueError(f"Invalid symbol record for {paper_id}:{equation_id}")
		canonical = symbol.get("canonical")
		definition = symbol.get("definition")
		if not isinstance(canonical, str) or not canonical:
			raise ValueError(f"Invalid canonical symbol for {paper_id}:{equation_id}")
		if canonical in seen:
			raise ValueError(
				f"Duplicate canonical symbol {canonical!r} for "
				f"{paper_id}:{equation_id}"
			)
		seen.add(canonical)
		if not isinstance(definition, str):
			raise ValueError(
				f"Invalid definition for {canonical!r} in {paper_id}:{equation_id}"
			)
		result[canonical] = definition
	return result


def _build_relations(
	entry: dict,
	paper_id: str,
	equation_id: str,
	expected_targets: set[str],
) -> dict[str, dict[str, str]]:
	relations = entry.get("relations")
	if not isinstance(relations, dict) or set(relations) != expected_targets:
		raise ValueError(
			f"Incomplete relations for {paper_id}:{equation_id}: "
			f"expected={sorted(expected_targets)}, "
			f"actual={sorted(relations) if isinstance(relations, dict) else None}"
		)
	result = {}
	for target_id in sorted(relations, key=_equation_sort_key):
		relation = relations[target_id]
		if not isinstance(relation, dict):
			raise ValueError(
				f"Invalid relation {paper_id}:{equation_id}->{target_id}"
			)
		grade = relation.get("grade")
		description = relation.get("description")
		if grade not in VALID_GRADES or not isinstance(description, str):
			raise ValueError(
				f"Invalid relation {paper_id}:{equation_id}->{target_id}"
			)
		if grade == "none" and description:
			raise ValueError(
				f"None relation must have an empty description: "
				f"{paper_id}:{equation_id}->{target_id}"
			)
		if grade != "none" and not description:
			raise ValueError(
				f"Retained relation requires a description: "
				f"{paper_id}:{equation_id}->{target_id}"
			)
		result[target_id] = {"grade": grade, "description": description}
	return result


def build_final_data(
	equations_file: Path,
	paper_list_file: Path,
	meanings_dir: Path,
	symbols_dir: Path,
	relations_dir: Path,
) -> dict[str, dict[str, dict]]:
	base = _load_json(equations_file)
	if not isinstance(base, dict):
		raise ValueError("Equation corpus must be a JSON object")
	for paper_id, equations in base.items():
		if not isinstance(paper_id, str) or not isinstance(equations, dict):
			raise ValueError("Invalid equation corpus structure")
	paper_order = _load_paper_order(paper_list_file, set(base))

	meanings = _validate_component_coverage(
		base, _load_component_directory(meanings_dir, "meaning"), "meaning"
	)
	symbols = _validate_component_coverage(
		base, _load_component_directory(symbols_dir, "symbol"), "symbol"
	)
	relations = _validate_component_coverage(
		base, _load_component_directory(relations_dir, "relation"), "relation"
	)

	result = {}
	for paper_id in paper_order:
		result[paper_id] = {}
		equation_ids = set(base[paper_id])
		for equation_id in sorted(equation_ids, key=_equation_sort_key):
			base_entry = base[paper_id][equation_id]
			if not isinstance(base_entry, dict):
				raise ValueError(f"Invalid base equation {paper_id}:{equation_id}")
			equation = base_entry.get("equation")
			meaning_entry = meanings[paper_id][equation_id]
			symbol_entry = symbols[paper_id][equation_id]
			relation_entry = relations[paper_id][equation_id]
			if not isinstance(equation, str):
				raise ValueError(f"Invalid equation text for {paper_id}:{equation_id}")
			if meaning_entry.get("equation") != equation:
				raise ValueError(f"Meaning equation mismatch for {paper_id}:{equation_id}")
			if symbol_entry.get("latex") != equation:
				raise ValueError(f"Symbol equation mismatch for {paper_id}:{equation_id}")
			meaning = meaning_entry.get("meaning")
			if not isinstance(meaning, str):
				raise ValueError(f"Invalid meaning for {paper_id}:{equation_id}")
			entry = {
				"equation": equation,
				"meaning": meaning,
				"symbols": _build_symbols(symbol_entry, paper_id, equation_id),
				"relations": _build_relations(
					relation_entry,
					paper_id,
					equation_id,
					equation_ids - {equation_id},
				),
			}
			if tuple(entry) != FINAL_FIELDS:
				raise AssertionError("Final equation fields are not in strict order")
			result[paper_id][equation_id] = entry
	return result


def export_final_data(
	equations_file: Path,
	paper_list_file: Path,
	meanings_dir: Path,
	symbols_dir: Path,
	relations_dir: Path,
	output_file: Path,
) -> tuple[int, int, int, int]:
	payload = build_final_data(
		equations_file, paper_list_file, meanings_dir, symbols_dir, relations_dir
	)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)
	equation_count = sum(len(equations) for equations in payload.values())
	symbol_count = sum(
		len(entry["symbols"])
		for equations in payload.values()
		for entry in equations.values()
	)
	relation_count = sum(
		len(entry["relations"])
		for equations in payload.values()
		for entry in equations.values()
	)
	return len(payload), equation_count, symbol_count, relation_count
