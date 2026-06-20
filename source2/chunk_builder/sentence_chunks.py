from index import PaperIndex, unique
from models import Chunk


def build_sentence_chunks(index: PaperIndex) -> list[Chunk]:
	chunks = []
	for section in index.paper["sections"]:
		metadata = index.section_metadata(section["section_id"])
		for paragraph in section["paragraphs"]:
			for sentence in paragraph["sentences"]:
				references = index.references_by_sentence.get(sentence["sentence_id"], [])
				nearby_equation_ids = unique([
					*paragraph["nearby_equation_ids"],
					*(
						equation_id
						for reference in references
						for equation_id in reference["target_equation_ids"]
					),
				])
				chunks.append(Chunk(
					chunk_id=f"{index.paper['paper_id']}:sentence:{sentence['sentence_id']}",
					paper_id=index.paper["paper_id"],
					chunk_type="sentence",
					text=sentence["text"],
					paragraph_ids=[paragraph["paragraph_id"]],
					sentence_ids=[sentence["sentence_id"]],
					nearby_equation_ids=nearby_equation_ids,
					source=index.paper["source_status"],
					**metadata,
				))
	return chunks
