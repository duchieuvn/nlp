from index import PaperIndex
from models import Chunk


def build_section_aware_chunks(index: PaperIndex) -> list[Chunk]:
	chunks = []
	for section in index.paper["sections"]:
		metadata = index.section_metadata(section["section_id"])
		for paragraph in section["paragraphs"]:
			text = paragraph["text"]
			if section["title"]:
				text = f"Section: {section['title']}\n{text}"
			chunks.append(Chunk(
				chunk_id=(
					f"{index.paper['paper_id']}:section_aware:"
					f"{paragraph['paragraph_id']}"
				),
				paper_id=index.paper["paper_id"],
				chunk_type="section_aware",
				text=text,
				paragraph_ids=[paragraph["paragraph_id"]],
				sentence_ids=[s["sentence_id"] for s in paragraph["sentences"]],
				nearby_equation_ids=paragraph["nearby_equation_ids"],
				source=index.paper["source_status"],
				**metadata,
			))
	return chunks
