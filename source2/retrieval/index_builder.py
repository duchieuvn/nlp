import json
from pathlib import Path

from retrieval_config import DEFAULT_CHUNK_TYPES
from retrieval_models import ChunkDocument


def load_chunk_documents(
	chunks_dir: Path,
	chunk_types: tuple[str, ...] | list[str] | None = DEFAULT_CHUNK_TYPES,
) -> list[ChunkDocument]:
	documents = []
	allowed_types = set(chunk_types) if chunk_types is not None else None
	for input_file in sorted(chunks_dir.glob("*.json")):
		payload = json.loads(input_file.read_text(encoding="utf-8"))
		if payload["paper_id"] != input_file.stem:
			raise ValueError(
				f"Chunk filename {input_file.name!r} does not match paper ID "
				f"{payload['paper_id']!r}"
			)
		for value in payload["chunks"]:
			if allowed_types is None or value["chunk_type"] in allowed_types:
				documents.append(ChunkDocument.from_dict(value))

	chunk_ids = [document.chunk_id for document in documents]
	if len(chunk_ids) != len(set(chunk_ids)):
		raise ValueError("Retrieval index contains duplicate chunk IDs")
	return documents
