import json
from pathlib import Path

from symbol_builder import build_paper_symbols
from chunk_enricher import enrich_chunk_file


def _write_json(payload: dict, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)


def build_symbol_files(
	equations_file: Path,
	output_dir: Path,
	chunks_dir: Path | None = None,
) -> tuple[int, int, int]:
	corpus = json.loads(equations_file.read_text(encoding="utf-8"))
	paper_count = 0
	symbol_count = 0
	enriched_chunk_count = 0
	for paper_id, equations in corpus.items():
		if not equations:
			continue
		payload = build_paper_symbols(paper_id, equations)
		_write_json(payload, output_dir / f"{paper_id}.json")
		paper_count += 1
		symbol_count += sum(
			len(equation["symbols"]) for equation in payload["equations"]
		)
		if chunks_dir:
			enriched_chunk_count += enrich_chunk_file(
				chunks_dir / f"{paper_id}.json",
				payload,
			)
	return paper_count, symbol_count, enriched_chunk_count
