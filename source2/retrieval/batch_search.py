import json
from pathlib import Path

from query_builder import build_equation_meaning_query
from retrieval_models import SearchQuery
from retrieval_service import RetrievalService


def _write_json(payload: dict, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)


def search_paper_equations(
	service: RetrievalService,
	symbol_payload: dict,
	method: str = "bm25",
	top_k: int = 10,
) -> dict:
	paper_id = symbol_payload["paper_id"]
	queries = []
	for equation in symbol_payload["equations"]:
		query_text = build_equation_meaning_query(equation)
		query = SearchQuery(
			text=query_text,
			paper_id=paper_id,
			chunk_types=["sentence", "paragraph", "equation_neighborhood"],
			equation_ids=[equation["equation_id"]],
			top_k=top_k,
		)
		queries.append({
			"task": "equation_meaning_evidence",
			"equation_id": equation["equation_id"],
			"query": query_text,
			"symbols": [
				symbol["canonical"] for symbol in equation.get("symbols", [])
			],
			"results": [
				result.to_dict() for result in service.search(query, method)
			],
		})
	return {
		"paper_id": paper_id,
		"method": method,
		"top_k": top_k,
		"queries": queries,
	}


def search_all_papers(
	service: RetrievalService,
	symbols_dir: Path,
	output_dir: Path,
	method: str = "bm25",
	top_k: int = 10,
) -> tuple[int, int, int]:
	paper_count = 0
	query_count = 0
	result_count = 0
	for symbol_file in sorted(symbols_dir.glob("*.json")):
		symbol_payload = json.loads(symbol_file.read_text(encoding="utf-8"))
		if symbol_payload["paper_id"] != symbol_file.stem:
			raise ValueError(
				f"Symbol filename {symbol_file.name!r} does not match paper ID "
				f"{symbol_payload['paper_id']!r}"
			)
		payload = search_paper_equations(
			service,
			symbol_payload,
			method=method,
			top_k=top_k,
		)
		_write_json(payload, output_dir / symbol_file.name)
		paper_count += 1
		query_count += len(payload["queries"])
		result_count += sum(len(query["results"]) for query in payload["queries"])
	return paper_count, query_count, result_count
