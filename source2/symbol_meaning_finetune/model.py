from collections import Counter
from dataclasses import dataclass
import random

from .config import (
	BATCH_SIZE,
	DEVICE,
	EPOCHS,
	LABELS,
	LEARNING_RATE,
	MAX_TOKENS,
	MODEL_NAME,
	RANDOM_SEED,
	WEIGHT_DECAY,
)
from .data import RelationExample


class EncodedRelationDataset:
	def __init__(self, examples: list[RelationExample], tokenizer):
		import torch

		self.examples = examples
		self.encodings = tokenizer(
			[example.symbol_context for example in examples],
			[example.candidate_context for example in examples],
			padding="max_length",
			truncation=True,
			max_length=MAX_TOKENS,
		)
		label_ids = {label: index for index, label in enumerate(LABELS)}
		self.labels = [label_ids[example.label] for example in examples]
		self._torch = torch

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> dict:
		item = {
			key: self._torch.tensor(values[index], dtype=self._torch.long)
			for key, values in self.encodings.items()
		}
		item["labels"] = self._torch.tensor(
			self.labels[index], dtype=self._torch.long
		)
		return item


@dataclass(frozen=True)
class Prediction:
	gold_label: str
	predicted_label: str
	probabilities: dict[str, float]
	has_modifiers: bool
	paper_id: str
	equation_id: str
	canonical: str
	phrase: str


def _require_training_stack():
	try:
		import torch
		from transformers import AutoModelForSequenceClassification, AutoTokenizer
	except ImportError as exc:
		raise RuntimeError(
			"Fine-tuning requires the torch and transformers packages"
		) from exc
	return torch, AutoModelForSequenceClassification, AutoTokenizer


def _model_inputs(batch: dict, device) -> tuple[dict, object]:
	labels = batch["labels"].to(device)
	inputs = {
		key: value.to(device)
		for key, value in batch.items()
		if key != "labels"
	}
	return inputs, labels


def select_device(torch):
	if DEVICE == "cpu":
		return torch.device("cpu")
	if DEVICE not in {"auto", "cuda"}:
		raise ValueError(f"Unsupported training device: {DEVICE}")
	if not torch.cuda.is_available():
		if DEVICE == "cuda":
			raise RuntimeError("DEVICE='cuda' but PyTorch cannot access CUDA")
		print("CUDA is unavailable; training on CPU")
		return torch.device("cpu")
	try:
		probe = torch.empty(1, device="cuda")
		del probe
		torch.cuda.synchronize()
		device = torch.device("cuda")
		print(f"Training on CUDA: {torch.cuda.get_device_name(device)}")
		return device
	except (RuntimeError, torch.AcceleratorError) as exc:
		if DEVICE == "cuda":
			raise RuntimeError(
				"DEVICE='cuda', but the CUDA device is busy or unavailable"
			) from exc
		print(
			"CUDA was detected but is busy or unavailable; falling back to CPU"
		)
		return torch.device("cpu")


def train_model(splits: dict[str, list[RelationExample]], checkpoint_dir):
	torch, AutoModelForSequenceClassification, AutoTokenizer = _require_training_stack()
	from torch.utils.data import DataLoader, WeightedRandomSampler

	random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(RANDOM_SEED)

	label_ids = {label: index for index, label in enumerate(LABELS)}
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
	model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_NAME,
		num_labels=len(LABELS),
		id2label={index: label for label, index in label_ids.items()},
		label2id=label_ids,
	)
	train_data = EncodedRelationDataset(splits["train"], tokenizer)
	validation_data = EncodedRelationDataset(splits["validation"], tokenizer)
	counts = Counter(train_data.labels)
	sample_weights = [1.0 / counts[label] for label in train_data.labels]
	sampler = WeightedRandomSampler(
		sample_weights, num_samples=len(sample_weights), replacement=True,
		generator=torch.Generator().manual_seed(RANDOM_SEED),
	)
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
	validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)

	device = select_device(torch)
	model.to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
	)
	loss_function = torch.nn.CrossEntropyLoss()
	best_validation_loss = float("inf")
	history = []
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	for epoch in range(EPOCHS):
		model.train()
		train_loss = 0.0
		for batch in train_loader:
			inputs, labels = _model_inputs(batch, device)
			optimizer.zero_grad(set_to_none=True)
			logits = model(**inputs).logits
			loss = loss_function(logits, labels)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			train_loss += loss.item()

		model.eval()
		validation_loss = 0.0
		with torch.inference_mode():
			for batch in validation_loader:
				inputs, labels = _model_inputs(batch, device)
				validation_loss += loss_function(model(**inputs).logits, labels).item()
		train_loss /= max(1, len(train_loader))
		validation_loss /= max(1, len(validation_loader))
		history.append({
			"epoch": epoch + 1,
			"train_loss": train_loss,
			"validation_loss": validation_loss,
		})
		print(
			f"Epoch {epoch + 1}/{EPOCHS}: train_loss={train_loss:.4f}, "
			f"validation_loss={validation_loss:.4f}"
		)
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			model.save_pretrained(checkpoint_dir)
			tokenizer.save_pretrained(checkpoint_dir)

	best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
	best_model.to(device).eval()
	return best_model, tokenizer, device, history


def predict(
	model,
	tokenizer,
	device,
	examples: list[RelationExample],
) -> list[Prediction]:
	import torch
	from torch.utils.data import DataLoader

	dataset = EncodedRelationDataset(examples, tokenizer)
	loader = DataLoader(dataset, batch_size=BATCH_SIZE)
	output = []
	offset = 0
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			inputs, _ = _model_inputs(batch, device)
			probability_rows = torch.softmax(model(**inputs).logits, dim=-1).cpu()
			for probabilities in probability_rows.tolist():
				example = examples[offset]
				predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
				output.append(Prediction(
					example.label,
					LABELS[predicted_index],
					dict(zip(LABELS, map(float, probabilities))),
					example.has_modifiers,
					example.paper_id,
					example.equation_id,
					example.canonical,
					example.phrase,
				))
				offset += 1
	return output
