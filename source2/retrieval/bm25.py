from collections import Counter
import math

from tokenizer import tokenize


class BM25Retriever:
	def __init__(
		self,
		texts: list[str],
		k1: float = 1.5,
		b: float = 0.75,
	) -> None:
		if k1 <= 0:
			raise ValueError("BM25 k1 must be positive")
		if not 0 <= b <= 1:
			raise ValueError("BM25 b must be between 0 and 1")
		self.k1 = k1
		self.b = b
		self.term_frequencies = [Counter(tokenize(text)) for text in texts]
		self.document_lengths = [sum(values.values()) for values in self.term_frequencies]
		self.average_document_length = (
			sum(self.document_lengths) / len(self.document_lengths)
			if self.document_lengths else 0.0
		)
		document_frequencies = Counter(
			term
			for frequencies in self.term_frequencies
			for term in frequencies
		)
		document_count = len(texts)
		self.inverse_document_frequencies = {
			term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
			for term, frequency in document_frequencies.items()
		}

	def score(
		self,
		query_text: str,
		candidate_indices: list[int],
	) -> list[tuple[int, float]]:
		query_frequencies = Counter(tokenize(query_text))
		scores = []
		for index in candidate_indices:
			frequencies = self.term_frequencies[index]
			document_length = self.document_lengths[index]
			normalization = 1 - self.b
			if self.average_document_length:
				normalization += self.b * document_length / self.average_document_length
			score = 0.0
			for term, query_frequency in query_frequencies.items():
				term_frequency = frequencies.get(term, 0)
				if not term_frequency:
					continue
				idf = self.inverse_document_frequencies.get(term, 0.0)
				score += query_frequency * idf * (
					term_frequency * (self.k1 + 1)
					/ (term_frequency + self.k1 * normalization)
				)
			if score > 0:
				scores.append((index, score))
		return scores
