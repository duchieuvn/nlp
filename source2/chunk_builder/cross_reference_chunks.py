from index import PaperIndex, unique
from models import Chunk


def build_cross_reference_chunks(index: PaperIndex) -> list[Chunk]:
	chunks = []
	for reference in index.paper["cross_references"]:
		sentence = index.sentences[reference["source_sentence_id"]]
		paragraph = index.paragraphs[reference["source_paragraph_id"]]
		chunks.append(Chunk(
			chunk_id=(
				f"{index.paper['paper_id']}:cross_reference:"
				f"{reference['reference_id']}"
			),
			paper_id=index.paper["paper_id"],
			chunk_type="cross_reference",
			text=sentence["text"],
			paragraph_ids=[paragraph["paragraph_id"]],
			sentence_ids=[sentence["sentence_id"]],
			nearby_equation_ids=unique([
				*paragraph["nearby_equation_ids"],
				*reference["target_equation_ids"],
			]),
			source=index.paper["source_status"],
			**index.section_metadata(reference["source_section_id"]),
		))
	return chunks
