import json
from pathlib import Path

from symbol_config import TOP_K
from symbol_extractor import extract_symbol_meaning
from symbol_retrieval import retrieve_symbol_evidence


def _write_json(payload: dict, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)


def extract_paper_symbol_meanings(
	service,
	symbol_payload: dict,
	top_k: int = TOP_K,
	nlp=None,
) -> dict:
	paper_id = symbol_payload["paper_id"]
	equations = []
	for equation in symbol_payload["equations"]:
		equation_id = equation["equation_id"]
		records = []
		for symbol in equation["symbols"]:
			query, results = retrieve_symbol_evidence(
				service, paper_id, equation_id, symbol, top_k
			)
			record = extract_symbol_meaning(
				symbol, equation_id, results, nlp=nlp
			).to_dict()
			record["audit"]["query"] = query
			records.append(record)
		equations.append({
			"equation_id": equation_id,
			"latex": equation["latex"],
			"symbols": records,
		})
	return {
		"paper_id": paper_id,
		"retrieval_method": "bm25",
		"top_k": top_k,
		"equations": equations,
	}


def extract_all_symbol_meanings(
	service,
	symbols_dir: Path,
	output_dir: Path,
	top_k: int = TOP_K,
	nlp=None,
) -> tuple[int, int, int]:
	paper_count = 0
	symbol_count = 0
	definition_count = 0
	for symbol_file in sorted(symbols_dir.glob("*.json")):
		symbol_payload = json.loads(symbol_file.read_text(encoding="utf-8"))
		if symbol_payload["paper_id"] != symbol_file.stem:
			raise ValueError(f"Invalid symbol paper: {symbol_file.name}")
		payload = extract_paper_symbol_meanings(
			service, symbol_payload, top_k, nlp=nlp
		)
		_write_json(payload, output_dir / symbol_file.name)
		paper_count += 1
		for equation in payload["equations"]:
			symbol_count += len(equation["symbols"])
			definition_count += sum(
				bool(symbol["definition"]) for symbol in equation["symbols"]
			)
	return paper_count, symbol_count, definition_count
