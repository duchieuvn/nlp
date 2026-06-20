from pathlib import Path

from bm25 import BM25Retriever
from filters import matching_indices
from index_builder import load_chunk_documents
from retrieval_models import ChunkDocument, SearchQuery, SearchResult
from tfidf import TfidfRetriever


class RetrievalService:
	def __init__(self, documents: list[ChunkDocument]) -> None:
		if not documents:
			raise ValueError("Retrieval index requires at least one chunk")
		self.documents = documents
		texts = [document.text for document in documents]
		self._bm25 = BM25Retriever(texts)
		self._tfidf = None

	@classmethod
	def from_directory(
		cls,
		chunks_dir: Path,
		chunk_types: tuple[str, ...] | list[str] | None = None,
	) -> "RetrievalService":
		if chunk_types is None:
			from retrieval_config import DEFAULT_CHUNK_TYPES

			chunk_types = DEFAULT_CHUNK_TYPES
		return cls(load_chunk_documents(chunks_dir, chunk_types))

	def search(
		self,
		query: SearchQuery,
		method: str = "bm25",
	) -> list[SearchResult]:
		candidate_indices = matching_indices(self.documents, query)
		if method == "bm25":
			scores = self._bm25.score(query.text, candidate_indices)
		elif method == "tfidf":
			if self._tfidf is None:
				self._tfidf = TfidfRetriever(
					[document.text for document in self.documents]
				)
			scores = self._tfidf.score(query.text, candidate_indices)
		else:
			raise ValueError(f"Unknown retrieval method: {method!r}")

		ranked = sorted(
			scores,
			key=lambda item: (-item[1], self.documents[item[0]].chunk_id),
		)[:query.top_k]
		return [
			self._result(rank, index, score, method)
			for rank, (index, score) in enumerate(ranked, start=1)
		]

	def _result(
		self,
		rank: int,
		index: int,
		score: float,
		method: str,
	) -> SearchResult:
		document = self.documents[index]
		return SearchResult(
			rank=rank,
			chunk_id=document.chunk_id,
			score=score,
			method=method,
			chunk_type=document.chunk_type,
			text=document.text,
			paper_id=document.paper_id,
			section_id=document.section_id,
			section_title=document.section_title,
			paragraph_ids=document.paragraph_ids,
			sentence_ids=document.sentence_ids,
			nearby_equation_ids=document.nearby_equation_ids,
			symbols=document.symbols,
			source=document.source,
		)
