import json
from pathlib import Path

from meaning_extractor import extract_equation_meaning


def _write_json(payload: dict, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)


def extract_paper_meanings(
	retrieval_payload: dict,
	equations: dict,
	reranker=None,
) -> dict:
	queries = {
		query["equation_id"]: query for query in retrieval_payload["queries"]
	}
	if set(queries) != set(equations):
		raise ValueError(
			f"Equation mismatch for paper {retrieval_payload['paper_id']}"
		)
	records = [
		extract_equation_meaning(
			queries[equation_id], entry["equation"], reranker=reranker
		)
		for equation_id, entry in equations.items()
	]
	return {
		"paper_id": retrieval_payload["paper_id"],
		"retrieval_method": retrieval_payload["method"],
		"equations": [record.to_dict() for record in records],
	}


def extract_all_meanings(
	retrieval_dir: Path,
	equations_file: Path,
	output_dir: Path,
	reranker=None,
) -> tuple[int, int, int]:
	corpus = json.loads(equations_file.read_text(encoding="utf-8"))
	paper_count = 0
	equation_count = 0
	meaning_count = 0
	for retrieval_file in sorted(retrieval_dir.glob("*.json")):
		retrieval_payload = json.loads(retrieval_file.read_text(encoding="utf-8"))
		paper_id = retrieval_payload["paper_id"]
		if paper_id != retrieval_file.stem or paper_id not in corpus:
			raise ValueError(f"Invalid retrieval paper: {retrieval_file.name}")
		payload = extract_paper_meanings(
			retrieval_payload, corpus[paper_id], reranker=reranker
		)
		_write_json(payload, output_dir / retrieval_file.name)
		paper_count += 1
		equation_count += len(payload["equations"])
		meaning_count += sum(bool(record["meaning"]) for record in payload["equations"])
	return paper_count, equation_count, meaning_count
