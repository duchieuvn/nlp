from index import PaperIndex, unique
from models import Chunk


def build_equation_neighborhood_chunks(index: PaperIndex) -> list[Chunk]:
	chunks = []
	for equation in index.paper["equations"]:
		paragraph_ids = unique([
			equation["previous_paragraph_id"],
			equation["next_paragraph_id"],
		])
		parts = []
		section = index.sections.get(equation["section_id"])
		if section and section["title"]:
			parts.append(f"Section: {section['title']}")
		if equation["previous_paragraph_id"] in index.paragraphs:
			parts.append(index.paragraphs[equation["previous_paragraph_id"]]["text"])
		parts.append(f"Equation ({equation['equation_id']}): {equation['latex']}")
		if equation["next_paragraph_id"] in index.paragraphs:
			parts.append(index.paragraphs[equation["next_paragraph_id"]]["text"])

		chunks.append(Chunk(
			chunk_id=(
				f"{index.paper['paper_id']}:equation_neighborhood:"
				f"{equation['equation_id']}"
			),
			paper_id=index.paper["paper_id"],
			chunk_type="equation_neighborhood",
			text="\n".join(parts),
			paragraph_ids=paragraph_ids,
			sentence_ids=index.sentence_ids(paragraph_ids),
			nearby_equation_ids=unique([
				equation["equation_id"],
				*index.nearby_equation_ids(paragraph_ids),
			]),
			source=index.paper["source_status"],
			**index.section_metadata(equation["section_id"]),
		))
	return chunks
