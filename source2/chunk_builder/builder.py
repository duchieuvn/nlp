from cross_reference_chunks import build_cross_reference_chunks
from equation_neighborhood_chunks import build_equation_neighborhood_chunks
from index import PaperIndex
from paragraph_chunks import build_paragraph_chunks
from section_aware_chunks import build_section_aware_chunks
from sentence_chunks import build_sentence_chunks
from validation import validate_chunks


def build_chunk_views(paper: dict) -> list[dict]:
	index = PaperIndex(paper)
	chunks = [
		*build_sentence_chunks(index),
		*build_paragraph_chunks(index),
		*build_equation_neighborhood_chunks(index),
		*build_section_aware_chunks(index),
		*build_cross_reference_chunks(index),
	]
	validate_chunks(chunks, paper)
	return [chunk.to_dict() for chunk in chunks]
