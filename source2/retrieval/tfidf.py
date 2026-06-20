from tokenizer import tokenize


class TfidfRetriever:
	def __init__(self, texts: list[str]) -> None:
		try:
			from sklearn.feature_extraction.text import TfidfVectorizer
		except ImportError as error:
			raise RuntimeError(
				"scikit-learn is required for TF-IDF retrieval"
			) from error
		self.vectorizer = TfidfVectorizer(
			tokenizer=tokenize,
			token_pattern=None,
			lowercase=False,
			ngram_range=(1, 2),
			norm="l2",
		)
		self.matrix = self.vectorizer.fit_transform(texts)

	def score(
		self,
		query_text: str,
		candidate_indices: list[int],
	) -> list[tuple[int, float]]:
		if not candidate_indices:
			return []
		query_vector = self.vectorizer.transform([query_text])
		similarities = self.matrix[candidate_indices].dot(query_vector.T).toarray().ravel()
		return [
			(index, float(score))
			for index, score in zip(candidate_indices, similarities)
			if score > 0
		]
