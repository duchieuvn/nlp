from models import Chunk


CHUNK_TYPES = {
	"sentence",
	"paragraph",
	"equation_neighborhood",
	"section_aware",
	"cross_reference",
}


def validate_chunks(chunks: list[Chunk], paper: dict) -> None:
	errors = []
	chunk_ids = [chunk.chunk_id for chunk in chunks]
	if len(chunk_ids) != len(set(chunk_ids)):
		errors.append("chunk IDs are not unique")

	section_ids = {section["section_id"] for section in paper["sections"]}
	paragraph_ids = {
		paragraph["paragraph_id"]
		for section in paper["sections"]
		for paragraph in section["paragraphs"]
	}
	sentence_ids = {
		sentence["sentence_id"]
		for section in paper["sections"]
		for paragraph in section["paragraphs"]
		for sentence in paragraph["sentences"]
	}
	equation_ids = {equation["equation_id"] for equation in paper["equations"]}

	for chunk in chunks:
		if chunk.paper_id != paper["paper_id"]:
			errors.append(f"{chunk.chunk_id}: incorrect paper ID")
		if chunk.chunk_type not in CHUNK_TYPES:
			errors.append(f"{chunk.chunk_id}: unknown chunk type")
		if not chunk.text.strip():
			errors.append(f"{chunk.chunk_id}: empty text")
		if chunk.section_id and chunk.section_id not in section_ids:
			errors.append(f"{chunk.chunk_id}: unknown section")
		if any(value not in paragraph_ids for value in chunk.paragraph_ids):
			errors.append(f"{chunk.chunk_id}: unknown paragraph")
		if any(value not in sentence_ids for value in chunk.sentence_ids):
			errors.append(f"{chunk.chunk_id}: unknown sentence")
		if any(value not in equation_ids for value in chunk.nearby_equation_ids):
			errors.append(f"{chunk.chunk_id}: unknown equation")

	if errors:
		raise ValueError("Chunk validation failed:\n- " + "\n- ".join(errors[:50]))
