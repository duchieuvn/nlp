import json
from pathlib import Path

from relation_builder import build_paper_relations


def _write_json(payload: dict, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)


def build_all_relation_files(
	structured_dir: Path,
	symbols_dir: Path,
	output_dir: Path,
) -> tuple[int, int, int]:
	paper_count = 0
	pair_count = 0
	strong_count = 0
	for document_file in sorted(structured_dir.glob("*.json")):
		symbol_file = symbols_dir / document_file.name
		if not symbol_file.exists():
			raise FileNotFoundError(f"Missing symbols for {document_file.name}")
		document = json.loads(document_file.read_text(encoding="utf-8"))
		symbol_payload = json.loads(symbol_file.read_text(encoding="utf-8"))
		payload = build_paper_relations(document, symbol_payload)
		_write_json(payload, output_dir / document_file.name)
		paper_count += 1
		for equation in payload["equations"]:
			pair_count += len(equation["relations"])
			strong_count += sum(
				relation["grade"] == "strong"
				for relation in equation["relations"].values()
			)
	return paper_count, pair_count, strong_count
