import json
from pathlib import Path

from builder import build_chunk_views


def build_chunk_file(input_file: Path, output_dir: Path) -> tuple[str, int]:
	paper = json.loads(input_file.read_text(encoding="utf-8"))
	paper_id = paper["paper_id"]
	if input_file.stem != paper_id:
		raise ValueError(
			f"Input filename {input_file.name!r} does not match paper ID {paper_id!r}"
		)
	payload = {
		"paper_id": paper_id,
		"title": paper["title"],
		"source_document": str(input_file),
		"chunks": build_chunk_views(paper),
	}
	output_dir.mkdir(parents=True, exist_ok=True)
	output_file = output_dir / f"{paper_id}.json"
	temporary_file = output_file.with_suffix(".json.tmp")
	temporary_file.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	temporary_file.replace(output_file)
	return paper_id, len(payload["chunks"])


def build_all_chunk_files(input_dir: Path, output_dir: Path) -> tuple[int, int]:
	files = sorted(input_dir.glob("*.json"))
	total_chunks = 0
	for input_file in files:
		_, chunk_count = build_chunk_file(input_file, output_dir)
		total_chunks += chunk_count
	return len(files), total_chunks
