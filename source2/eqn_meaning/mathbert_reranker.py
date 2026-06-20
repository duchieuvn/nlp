from __future__ import annotations

from typing import Any, Sequence

from candidates import MeaningCandidate
from meaning_config import (
	MATHBERT_BATCH_SIZE,
	MATHBERT_MAX_TOKENS,
	MATHBERT_MODEL_NAME,
)


class MathBERTReranker:
	"""Score extractive candidates with a shared MathBERT encoder."""

	def __init__(
		self,
		model_name: str = MATHBERT_MODEL_NAME,
		max_tokens: int = MATHBERT_MAX_TOKENS,
		batch_size: int = MATHBERT_BATCH_SIZE,
	) -> None:
		try:
			import torch
			from transformers import AutoModel, AutoTokenizer
		except ImportError as exc:
			raise RuntimeError(
				"MathBERT reranking requires the 'torch' and 'transformers' packages"
			) from exc

		self.model_name = model_name
		self.max_tokens = max_tokens
		self.batch_size = batch_size
		self._torch = torch
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
		self._model = AutoModel.from_pretrained(model_name).to(self.device)
		self._model.eval()

	def _embed(self, texts: Sequence[str]) -> Any:
		batches = []
		for start in range(0, len(texts), self.batch_size):
			encoded = self._tokenizer(
				list(texts[start:start + self.batch_size]),
				padding=True,
				truncation=True,
				max_length=self.max_tokens,
				return_tensors="pt",
			)
			encoded = {key: value.to(self.device) for key, value in encoded.items()}
			with self._torch.inference_mode():
				hidden = self._model(**encoded).last_hidden_state
			mask = encoded["attention_mask"].unsqueeze(-1)
			pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
			batches.append(self._torch.nn.functional.normalize(pooled, p=2, dim=1))
		return self._torch.cat(batches, dim=0)

	@staticmethod
	def _equation_representation(
		equation: str,
		query: dict,
		candidates: Sequence[MeaningCandidate],
	) -> str:
		symbols = query.get("symbols", [])
		if not symbols:
			symbols = [
				symbol
				for candidate in candidates
				for symbol in candidate.result.get("symbols", [])
			]
		sections = [
			candidate.result.get("section_title", "")
			for candidate in candidates
		]
		return " | ".join(filter(None, (
			f"mathematical equation: {equation}",
			f"symbols: {' '.join(dict.fromkeys(symbols))}" if symbols else "",
			f"section: {'; '.join(dict.fromkeys(filter(None, sections)))}"
			if sections else "",
		)))

	def score_candidates(
		self,
		equation: str,
		query: dict,
		candidates: Sequence[MeaningCandidate],
	) -> list[float]:
		if not candidates:
			return []
		reference = self._equation_representation(equation, query, candidates)
		texts = [reference, *(candidate.text for candidate in candidates)]
		embeddings = self._embed(texts)
		raw_scores = embeddings[1:] @ embeddings[0]
		return [float(score) for score in raw_scores.detach().cpu().tolist()]
