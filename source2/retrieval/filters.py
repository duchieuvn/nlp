from retrieval_models import ChunkDocument, SearchQuery


def _intersects(required: list[str] | None, actual: list[str]) -> bool:
	return required is None or bool(set(required) & set(actual))


def matching_indices(
	documents: list[ChunkDocument],
	query: SearchQuery,
) -> list[int]:
	return [
		index
		for index, document in enumerate(documents)
		if document.paper_id == query.paper_id
		and (
			query.section_ids is None
			or document.section_id in query.section_ids
		)
		and (
			query.chunk_types is None
			or document.chunk_type in query.chunk_types
		)
		and _intersects(query.equation_ids, document.nearby_equation_ids)
		and _intersects(query.symbols, document.symbols)
	]
