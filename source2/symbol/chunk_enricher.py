import json
from pathlib import Path


def enrich_chunk_file(chunk_file: Path, symbol_payload: dict) -> int:
	if not chunk_file.exists():
		return 0
	equation_symbols = {
		equation["equation_id"]: [
			symbol["canonical"] for symbol in equation["symbols"]
		]
		for equation in symbol_payload["equations"]
	}
	payload = json.loads(chunk_file.read_text(encoding="utf-8"))
	for chunk in payload["chunks"]:
		chunk["symbols"] = list(dict.fromkeys(
			symbol
			for equation_id in chunk["nearby_equation_ids"]
			for symbol in equation_symbols.get(equation_id, [])
		))
	temporary_file = chunk_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(chunk_file)
	return sum(bool(chunk["symbols"]) for chunk in payload["chunks"])
