import json

from .symbol_config import (
	CHECKPOINT_DIR,
	DEFAULT_RELATION_MARGIN,
	DEFAULT_RELATION_THRESHOLD,
	DEVICE,
	MAX_TOKENS,
	MODEL_BATCH_SIZE,
	REQUIRE_REVIEWED_CALIBRATION,
)
from .symbol_models import ParsedSymbol, PhraseCandidate, RELATION_LABELS


def symbol_context(parsed: ParsedSymbol, sentence: str) -> str:
	return " | ".join((
		f"original LaTeX: {parsed.original_latex}",
		f"canonical: {parsed.canonical}",
		f"equation: {parsed.equation}",
		f"context: {sentence}",
	))


def candidate_context(candidate: PhraseCandidate) -> str:
	return (
		f"candidate phrase: {candidate.phrase} | evidence: {candidate.sentence}"
	)


def _select_device(torch):
	if DEVICE == "cpu":
		return torch.device("cpu")
	if DEVICE not in {"auto", "cuda"}:
		raise ValueError(f"Unsupported inference device: {DEVICE}")
	if not torch.cuda.is_available():
		if DEVICE == "cuda":
			raise RuntimeError("DEVICE='cuda' but CUDA is unavailable")
		return torch.device("cpu")
	try:
		probe = torch.empty(1, device="cuda")
		del probe
		torch.cuda.synchronize()
		return torch.device("cuda")
	except RuntimeError as exc:
		if DEVICE == "cuda":
			raise RuntimeError("CUDA is busy or unavailable") from exc
		print("CUDA is busy or unavailable; using CPU for symbol inference")
		return torch.device("cpu")


class MathBERTCrossEncoder:
	def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
		try:
			import torch
			from transformers import AutoModelForSequenceClassification, AutoTokenizer
		except ImportError as exc:
			raise RuntimeError(
				"Checkpoint inference requires torch and transformers"
			) from exc
		if not (checkpoint_dir / "config.json").is_file():
			raise FileNotFoundError(f"Missing MathBERT checkpoint: {checkpoint_dir}")
		self.checkpoint_dir = checkpoint_dir
		self.model_name = str(checkpoint_dir)
		self._torch = torch
		self.device = _select_device(torch)
		self.tokenizer = AutoTokenizer.from_pretrained(
			checkpoint_dir, use_fast=True, local_files_only=True
		)
		self.model = AutoModelForSequenceClassification.from_pretrained(
			checkpoint_dir, local_files_only=True
		)
		try:
			self.model = self.model.to(self.device).eval()
		except RuntimeError as exc:
			if DEVICE == "cuda" or self.device.type != "cuda":
				raise
			print("Insufficient CUDA memory for checkpoint; reloading on CPU")
			del self.model
			torch.cuda.empty_cache()
			self.device = torch.device("cpu")
			self.model = AutoModelForSequenceClassification.from_pretrained(
				checkpoint_dir, local_files_only=True
			).eval()
		self.labels = [
			self.model.config.id2label[index]
			for index in range(self.model.config.num_labels)
		]
		unknown = set(self.labels) - set(RELATION_LABELS)
		if unknown:
			raise ValueError(f"Checkpoint has unknown relation labels: {sorted(unknown)}")
		self.calibration = self._load_calibration()

	def _load_calibration(self) -> dict:
		path = self.checkpoint_dir / "inference_config.json"
		payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
		scope = payload.get("calibration_scope", "missing calibration metadata")
		reviewed = "human-reviewed" in scope.casefold() and "not human-reviewed" not in scope.casefold()
		return {
			"threshold": float(payload.get("relation_threshold", DEFAULT_RELATION_THRESHOLD)),
			"margin": float(payload.get("relation_margin", DEFAULT_RELATION_MARGIN)),
			"scope": scope,
			"reviewed": reviewed,
			"acceptance_enabled": reviewed or not REQUIRE_REVIEWED_CALIBRATION,
		}

	def predict(
		self,
		parsed: ParsedSymbol,
		candidates: list[PhraseCandidate],
	) -> list[dict[str, float]]:
		output = []
		for start in range(0, len(candidates), MODEL_BATCH_SIZE):
			batch = candidates[start:start + MODEL_BATCH_SIZE]
			encoded = self.tokenizer(
				[symbol_context(parsed, candidate.sentence) for candidate in batch],
				[candidate_context(candidate) for candidate in batch],
				padding=True,
				truncation=True,
				max_length=MAX_TOKENS,
				return_tensors="pt",
			)
			encoded = {key: value.to(self.device) for key, value in encoded.items()}
			with self._torch.inference_mode():
				rows = self._torch.softmax(
					self.model(**encoded).logits, dim=-1
				).cpu().tolist()
			output.extend(dict(zip(self.labels, map(float, row))) for row in rows)
		return output
