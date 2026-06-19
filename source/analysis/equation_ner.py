from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import random
import re
from typing import Any, Iterable


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ANALYSIS_DIR.parents[1]
DEFAULT_ANALYSIS_FILE = ANALYSIS_DIR / "analysis_meaning.json"
DEFAULT_EQUATIONS_FILE = PROJECT_DIR / "data" / "3_equations.json"
DEFAULT_DATA_DIR = ANALYSIS_DIR / "ner_data"
DEFAULT_MODEL_DIR = ANALYSIS_DIR / "scibert-equation-ner-final"
DEFAULT_BASE_MODEL = "allenai/scibert_scivocab_uncased"
DEFAULT_MATH_MODEL = "witiko/mathberta"
DEFAULT_MAX_TOKENS = 256
DEFAULT_EQUATION_TOKENS = 32
EQUATION_MARKER = "[EQUATION]"
CONTEXT_MODES = ("marker", "formula")
LABELS = ("O", "B-EQ_NAME", "I-EQ_NAME")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


@dataclass(frozen=True)
class NerExample:
    paper_id: str
    equation_id: str
    tokens: list[str]
    ner_tags: list[str]
    meaning: str
    window: str
    equation: str = ""
    has_answer: bool = True
    start: int | None = None
    end: int | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_text(text: str) -> list[str]:
    token_pattern = re.compile(
        rf"{re.escape(EQUATION_MARKER)}"
        r"|\\[A-Za-z]+"
        r"|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
        r"|[^\s]"
    )
    return token_pattern.findall(normalize_text(text))


def normalized_match_token(token: str) -> str:
    token = token.lower()
    if token.startswith("\\"):
        token = token[1:]
    return re.sub(r"[^a-z0-9]+", "", token)


def normalized_token_sequence(tokens: list[str]) -> list[str]:
    return [
        normalized
        for token in tokens
        if (normalized := normalized_match_token(token))
    ]


def raw_to_normalized_positions(tokens: list[str]) -> tuple[list[str], list[int]]:
    normalized_tokens = []
    raw_positions = []
    for raw_index, token in enumerate(tokens):
        normalized = normalized_match_token(token)
        if normalized:
            normalized_tokens.append(normalized)
            raw_positions.append(raw_index)
    return normalized_tokens, raw_positions


def find_subsequence(haystack: list[str], needle: list[str]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def crop_labeled_tokens(
    tokens: list[str],
    tags: list[str],
    max_tokens: int,
) -> tuple[list[str], list[str]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if len(tokens) <= max_tokens:
        return tokens, tags

    labeled = [index for index, tag in enumerate(tags) if tag != "O"]
    markers = [index for index, token in enumerate(tokens) if token == EQUATION_MARKER]
    important = labeled + markers
    if not important:
        return tokens[:max_tokens], tags[:max_tokens]

    required_start = min(important)
    required_end = max(important) + 1
    if required_end - required_start >= max_tokens:
        start = required_start
    else:
        start = max(0, required_start - (max_tokens - required_end + required_start) // 2)
    start = min(start, len(tokens) - max_tokens)
    return tokens[start : start + max_tokens], tags[start : start + max_tokens]


def label_window_tokens(
    window: str,
    meaning: str,
    max_tokens: int,
    has_answer: bool = True,
) -> tuple[list[str], list[str]] | None:
    tokens = tokenize_text(window)
    if not has_answer:
        return crop_labeled_tokens(tokens, ["O"] * len(tokens), max_tokens)

    meaning_tokens = tokenize_text(meaning)
    normalized_window, raw_positions = raw_to_normalized_positions(tokens)
    normalized_meaning = normalized_token_sequence(meaning_tokens)
    normalized_start = find_subsequence(normalized_window, normalized_meaning)
    if normalized_start is None:
        return None

    normalized_end = normalized_start + len(normalized_meaning)
    raw_start = raw_positions[normalized_start]
    raw_end = raw_positions[normalized_end - 1] + 1
    tags = ["O"] * len(tokens)
    tags[raw_start] = "B-EQ_NAME"
    for index in range(raw_start + 1, raw_end):
        tags[index] = "I-EQ_NAME"
    return crop_labeled_tokens(tokens, tags, max_tokens)


def validate_reviewed_record(payload: dict[str, Any]) -> str:
    if payload.get("review_status") != "reviewed":
        return "record has not been reviewed"
    has_answer = payload.get("has_answer")
    if not isinstance(has_answer, bool):
        return "has_answer must be true or false"
    window = payload.get("window", "")
    if not isinstance(window, str) or not window:
        return "window is empty"
    if not has_answer:
        if payload.get("meaning") or payload.get("start") is not None or payload.get("end") is not None:
            return "no-answer records must have an empty meaning and null offsets"
        return ""

    meaning = payload.get("meaning", "")
    start = payload.get("start")
    end = payload.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return "answer offsets must be integers"
    if not (0 <= start < end <= len(window)):
        return "answer offsets are outside the window"
    if window[start:end] != meaning:
        return "meaning does not exactly match the source offsets"
    return ""


def load_reviewed_records(analysis_file: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(analysis_file.read_text(encoding="utf-8"))
    records = []
    rejected = []
    for paper_id, equations in data.items():
        for equation_id, payload in equations.items():
            reason = validate_reviewed_record(payload)
            identity = {"paper_id": paper_id, "equation_id": equation_id}
            if reason:
                rejected.append({**identity, "reason": reason})
                continue
            records.append({**identity, **payload})
    return records, rejected


def iter_labeled_records(analysis_file: Path) -> list[dict[str, Any]]:
    """Return validated reviewed records for compatibility with the original API."""
    records, _ = load_reviewed_records(analysis_file)
    return records


def build_examples(
    records: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[list[NerExample], list[dict[str, Any]]]:
    examples = []
    skipped = []
    for record in records:
        labeled = label_window_tokens(
            record["window"],
            record.get("meaning", ""),
            max_tokens,
            record["has_answer"],
        )
        if labeled is None:
            skipped.append({
                "paper_id": record["paper_id"],
                "equation_id": record["equation_id"],
                "meaning": record.get("meaning", ""),
                "reason": "meaning tokens were not found as a contiguous span in window",
            })
            continue
        tokens, tags = labeled
        examples.append(NerExample(
            paper_id=record["paper_id"],
            equation_id=record["equation_id"],
            tokens=tokens,
            ner_tags=tags,
            meaning=record.get("meaning", ""),
            window=record["window"],
            equation=record.get("equation", ""),
            has_answer=record["has_answer"],
            start=record.get("start"),
            end=record.get("end"),
        ))
    return examples, skipped


def split_examples(
    examples: list[NerExample],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, list[NerExample]]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("ratios must leave non-empty proportions for train and test")
    paper_ids = sorted({example.paper_id for example in examples})
    random.Random(seed).shuffle(paper_ids)
    train_end = int(len(paper_ids) * train_ratio)
    validation_end = train_end + int(len(paper_ids) * validation_ratio)
    paper_split = {
        paper_id: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, paper_id in enumerate(paper_ids)
    }
    splits = {"train": [], "validation": [], "test": []}
    for example in examples:
        splits[paper_split[example.paper_id]].append(example)
    return splits


def example_to_json(example: NerExample) -> dict[str, Any]:
    return {
        "paper_id": example.paper_id,
        "equation_id": example.equation_id,
        "tokens": example.tokens,
        "ner_tags": example.ner_tags,
        "meaning": example.meaning,
        "window": example.window,
        "equation": example.equation,
        "has_answer": example.has_answer,
        "start": example.start,
        "end": example.end,
    }


def write_jsonl(path: Path, examples: Iterable[NerExample]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example_to_json(example), ensure_ascii=False) + "\n")


def build_dataset(
    analysis_file: Path = DEFAULT_ANALYSIS_FILE,
    output_dir: Path = DEFAULT_DATA_DIR,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 13,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    records, rejected = load_reviewed_records(analysis_file)
    examples, skipped = build_examples(records, max_tokens)
    splits = split_examples(examples, train_ratio, validation_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records)

    report = {
        "source_file": str(analysis_file),
        "output_dir": str(output_dir),
        "label_set": list(LABELS),
        "max_tokens": max_tokens,
        "reviewed_records": len(records),
        "unreviewed_or_invalid_records": len(rejected),
        "usable_examples": len(examples),
        "answerable_examples": sum(example.has_answer for example in examples),
        "no_answer_examples": sum(not example.has_answer for example in examples),
        "skipped_examples": len(skipped),
        "splits": {name: len(items) for name, items in splits.items()},
        "split_papers": {name: len({item.paper_id for item in items}) for name, items in splits.items()},
        "rejected": rejected,
        "skipped": skipped,
        "seed": seed,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def find_exact_span(window: str, meaning: str, prefer_before_marker: bool = True) -> tuple[int, int] | None:
    if not window or not meaning:
        return None
    marker = window.find(EQUATION_MARKER)
    positions = [match.start() for match in re.finditer(re.escape(meaning), window)]
    if not positions:
        return None
    if prefer_before_marker and marker >= 0:
        before = [position for position in positions if position < marker]
        if before:
            start = before[-1]
            return start, start + len(meaning)
    start = positions[0]
    return start, start + len(meaning)


def export_review_template(
    equations_file: Path = DEFAULT_EQUATIONS_FILE,
    existing_file: Path = DEFAULT_ANALYSIS_FILE,
    output_file: Path = DEFAULT_ANALYSIS_FILE,
) -> dict[str, int]:
    equations = json.loads(equations_file.read_text(encoding="utf-8"))
    existing = json.loads(existing_file.read_text(encoding="utf-8")) if existing_file.exists() else {}
    output: dict[str, dict[str, Any]] = {}
    counts = {"total": 0, "reviewed": 0, "needs_review": 0}
    for paper_id, paper_equations in equations.items():
        output[paper_id] = {}
        for equation_id, entry in paper_equations.items():
            counts["total"] += 1
            old = existing.get(paper_id, {}).get(equation_id, {})
            window = entry.get("surrounding_text", {}).get("window", "")
            meaning = old.get("meaning", "")
            span = find_exact_span(window, meaning)
            reviewed = old.get("review_status") == "reviewed" or bool(meaning and span)
            has_answer: bool | None = bool(meaning) if reviewed else None
            payload = {
                "meaning": window[span[0] : span[1]] if span else "",
                "window": window,
                "equation": entry.get("equation", ""),
                "has_answer": has_answer,
                "start": span[0] if span else None,
                "end": span[1] if span else None,
                "review_status": "reviewed" if reviewed else "needs_review",
            }
            output[paper_id][equation_id] = payload
            counts[payload["review_status"]] += 1
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def equation_preview(equation: str, tokenizer: Any, max_equation_tokens: int = DEFAULT_EQUATION_TOKENS) -> str:
    if not equation or max_equation_tokens <= 0:
        return ""
    token_ids = tokenizer.encode(equation, add_special_tokens=False)[:max_equation_tokens]
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


def format_model_context(
    window: str,
    equation: str,
    tokenizer: Any,
    context_mode: str = "marker",
    max_equation_tokens: int = DEFAULT_EQUATION_TOKENS,
) -> str:
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context mode: {context_mode}")
    if context_mode == "marker" or EQUATION_MARKER not in window:
        return window
    preview = equation_preview(equation, tokenizer, max_equation_tokens)
    if not preview:
        return window
    return window.replace(EQUATION_MARKER, f"{EQUATION_MARKER} {preview}", 1)


class EquationNerDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        context_mode: str = "marker",
        max_equation_tokens: int = DEFAULT_EQUATION_TOKENS,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.context_mode = context_mode
        self.max_equation_tokens = max_equation_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tokens = list(record["tokens"])
        tags = list(record["ner_tags"])
        if self.context_mode == "formula" and EQUATION_MARKER in tokens:
            preview = equation_preview(record.get("equation", ""), self.tokenizer, self.max_equation_tokens)
            preview_tokens = tokenize_text(preview)
            marker_index = tokens.index(EQUATION_MARKER) + 1
            tokens[marker_index:marker_index] = preview_tokens
            tags[marker_index:marker_index] = ["O"] * len(preview_tokens)

        tokenized = self.tokenizer(tokens, truncation=True, is_split_into_words=True)
        label_ids = []
        previous_word_index = None
        for word_index in tokenized.word_ids():
            if word_index is None:
                label_ids.append(-100)
            elif word_index != previous_word_index:
                label_ids.append(LABEL2ID[tags[word_index]])
            else:
                label_ids.append(-100)
            previous_word_index = word_index
        tokenized["labels"] = label_ids
        return tokenized


def load_training_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from transformers import AutoModelForTokenClassification
        from transformers import AutoTokenizer
        from transformers import DataCollatorForTokenClassification
        from transformers import Trainer
        from transformers import TrainingArguments
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training requires transformers, torch, and their dependencies.") from exc
    return AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments


def train_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: float = 5.0,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    context_mode: str = "marker",
    seed: int = 13,
    runs_dir: Path | None = None,
) -> None:
    dependencies = load_training_dependencies()
    AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments = dependencies
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [EQUATION_MARKER]})
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    train_records = read_jsonl(data_dir / "train.jsonl")
    validation_records = read_jsonl(data_dir / "validation.jsonl")
    train_dataset = EquationNerDataset(train_records, tokenizer, context_mode)
    validation_dataset = EquationNerDataset(validation_records, tokenizer, context_mode)
    runs_dir = runs_dir or model_dir.parent / f"{model_dir.name}-runs"
    training_args = TrainingArguments(
        output_dir=str(runs_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (model_dir / "equation_ner_metadata.json").write_text(json.dumps({
        "base_model": base_model,
        "context_mode": context_mode,
        "seed": seed,
        "label_set": list(LABELS),
    }, indent=2) + "\n", encoding="utf-8")


def load_inference_pipeline(model_dir: Path) -> Any:
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError("Inference requires transformers and torch.") from exc
    return pipeline("ner", model=str(model_dir), tokenizer=str(model_dir), aggregation_strategy="none")


def _entity_label(entity: dict[str, Any]) -> str:
    return str(entity.get("entity", entity.get("entity_group", ""))).upper()


def source_spans(entities: list[dict[str, Any]], source_text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for entity in sorted(entities, key=lambda item: (int(item.get("start", -1)), int(item.get("end", -1)))):
        label = _entity_label(entity)
        if label not in {"B-EQ_NAME", "I-EQ_NAME", "EQ_NAME"}:
            continue
        start = int(entity.get("start", -1))
        end = int(entity.get("end", -1))
        if not (0 <= start < end <= len(source_text)):
            continue
        begins = label == "B-EQ_NAME"
        separated = current and start > int(current[-1]["end"]) + 1
        if current and (begins or separated):
            spans.append(_finish_span(current, source_text))
            current = []
        current.append({**entity, "start": start, "end": end})
    if current:
        spans.append(_finish_span(current, source_text))
    return spans


def _finish_span(entities: list[dict[str, Any]], source_text: str) -> dict[str, Any]:
    start = int(entities[0]["start"])
    end = int(entities[-1]["end"])
    return {
        "text": source_text[start:end],
        "start": start,
        "end": end,
        "confidence": sum(float(entity.get("score", 0.0)) for entity in entities) / len(entities),
    }


def predict_meaning(
    window: str,
    ner_pipeline: Any,
    equation: str = "",
    context_mode: str = "marker",
    model: str = "",
    threshold: float = 0.0,
) -> dict[str, Any]:
    source_text = format_model_context(window, equation, ner_pipeline.tokenizer, context_mode)
    entities = ner_pipeline(source_text)
    spans = source_spans(entities, source_text)
    if not spans:
        return {
            "meaning": "", "candidate": "", "start": None, "end": None, "confidence": 0.0,
            "status": "no_answer", "model": model, "context_mode": context_mode,
            "source_text": source_text, "entities": entities,
        }
    best = max(spans, key=lambda span: span["confidence"])
    if best["confidence"] < threshold:
        return {
            "meaning": "", "candidate": best["text"], "start": best["start"],
            "end": best["end"], "confidence": best["confidence"],
            "status": "rejected_low_confidence", "model": model,
            "context_mode": context_mode, "source_text": source_text, "entities": entities,
        }
    return {
        "meaning": best["text"], "candidate": best["text"], "start": best["start"], "end": best["end"],
        "confidence": best["confidence"], "status": "accepted", "model": model,
        "context_mode": context_mode, "source_text": source_text, "entities": entities,
    }


def _load_model_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "equation_ner_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def predict_file(
    input_file: Path = DEFAULT_ANALYSIS_FILE,
    output_file: Path = ANALYSIS_DIR / "ner_predictions.json",
    model_dir: Path = DEFAULT_MODEL_DIR,
    context_mode: str | None = None,
    threshold: float = 0.0,
) -> dict[str, Any]:
    ner_pipeline = load_inference_pipeline(model_dir)
    metadata = _load_model_metadata(model_dir)
    context_mode = context_mode or metadata.get("context_mode", "marker")
    data = json.loads(input_file.read_text(encoding="utf-8"))
    predictions: dict[str, dict[str, Any]] = {}
    filled = 0
    total = 0
    for paper_id, equations in data.items():
        predictions[paper_id] = {}
        for equation_id, payload in equations.items():
            total += 1
            result = predict_meaning(
                payload.get("window", ""), ner_pipeline, payload.get("equation", ""),
                context_mode, str(model_dir), threshold,
            )
            filled += bool(result["meaning"])
            predictions[paper_id][equation_id] = {
                "gold_meaning": payload.get("meaning", ""),
                "gold_has_answer": payload.get("has_answer"),
                **{key: result[key] for key in (
                    "meaning", "candidate", "start", "end", "confidence", "status", "model",
                    "context_mode", "source_text",
                )},
            }
    summary = {
        "model_dir": str(model_dir), "context_mode": context_mode, "threshold": threshold,
        "total": total, "filled": filled, "predictions": predictions,
    }
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _prediction_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "predictions" in data:
        data = data["predictions"]
    result = {}
    for paper_id, equations in data.items():
        for equation_id, payload in equations.items():
            if "surrounding_text" in payload:
                payload = {"meaning": payload.get("meaning", ""), "status": "accepted" if payload.get("meaning") else "no_answer"}
            result[(paper_id, equation_id)] = payload
    return result


def _answer_tokens(text: str) -> set[str]:
    return {token for token in normalized_token_sequence(tokenize_text(text)) if token}


def evaluate_predictions(gold_records: list[dict[str, Any]], predictions: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = exact = rejected = 0
    token_f1_total = 0.0
    for gold in gold_records:
        prediction = predictions.get((gold["paper_id"], gold["equation_id"]), {})
        predicted = normalize_text(prediction.get("meaning", prediction.get("predicted_meaning", "")))
        has_prediction = bool(predicted)
        has_answer = gold["has_answer"]
        true_positive += has_answer and has_prediction
        false_positive += not has_answer and has_prediction
        false_negative += has_answer and not has_prediction
        true_negative += not has_answer and not has_prediction
        rejected += str(prediction.get("status", "")).startswith("rejected")
        if has_answer and has_prediction:
            exact += predicted == gold["meaning"]
            gold_tokens = _answer_tokens(gold["meaning"])
            predicted_tokens = _answer_tokens(predicted)
            overlap = len(gold_tokens & predicted_tokens)
            precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
            recall = overlap / len(gold_tokens) if gold_tokens else 0.0
            token_f1_total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(gold_records)
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    return {
        "examples": total,
        "exact_match": exact / actual_positive if actual_positive else 0.0,
        "token_f1": token_f1_total / actual_positive if actual_positive else 0.0,
        "answer_precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "answer_recall": true_positive / actual_positive if actual_positive else 0.0,
        "no_answer_accuracy": true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0,
        "coverage": predicted_positive / total if total else 0.0,
        "rejected_span_rate": rejected / total if total else 0.0,
        "confusion": {"tp": true_positive, "fp": false_positive, "fn": false_negative, "tn": true_negative},
    }


def evaluate_files(test_file: Path, named_files: list[str], output_file: Path) -> dict[str, Any]:
    gold = read_jsonl(test_file)
    report = {"test_file": str(test_file), "examples": len(gold), "models": {}}
    for specification in named_files:
        if "=" not in specification:
            raise ValueError("predictions must use NAME=PATH")
        name, raw_path = specification.split("=", 1)
        report["models"][name] = evaluate_predictions(gold, _prediction_map(Path(raw_path)))
    output_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def tune_threshold(validation_file: Path, prediction_file: Path, output_file: Path) -> dict[str, Any]:
    gold = read_jsonl(validation_file)
    predictions = _prediction_map(prediction_file)
    confidences = sorted({
        float(payload.get("confidence", 0.0))
        for payload in predictions.values()
        if payload.get("meaning", payload.get("predicted_meaning", ""))
    })
    candidates = sorted({0.0, *confidences, 1.0})
    trials = []
    for threshold in candidates:
        filtered = {}
        for identity, payload in predictions.items():
            filtered[identity] = dict(payload)
            if float(payload.get("confidence", 0.0)) < threshold:
                filtered[identity]["meaning"] = ""
                filtered[identity]["predicted_meaning"] = ""
                filtered[identity]["status"] = "rejected_low_confidence"
        metrics = evaluate_predictions(gold, filtered)
        trials.append({"threshold": threshold, "metrics": metrics})
    best = max(
        trials,
        key=lambda trial: (
            trial["metrics"]["token_f1"],
            trial["metrics"]["answer_precision"],
            trial["metrics"]["answer_recall"],
            -trial["threshold"],
        ),
    )
    report = {
        "validation_file": str(validation_file),
        "prediction_file": str(prediction_file),
        "selection_metric": "token_f1, then precision, then recall",
        "recommended_threshold": best["threshold"],
        "recommended_metrics": best["metrics"],
        "trials": trials,
    }
    output_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build, train, predict, and evaluate equation-name NER models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-data")
    build.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    build.add_argument("--train-ratio", type=float, default=0.8)
    build.add_argument("--validation-ratio", type=float, default=0.1)
    build.add_argument("--seed", type=int, default=13)
    build.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    template = subparsers.add_parser("export-review-template")
    template.add_argument("--equations-file", type=Path, default=DEFAULT_EQUATIONS_FILE)
    template.add_argument("--existing-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    template.add_argument("--output-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    train = subparsers.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--runs-dir", type=Path)
    train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train.add_argument("--context-mode", choices=CONTEXT_MODES, default="marker")
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--epochs", type=float, default=5.0)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    predict.add_argument("--output-file", type=Path, default=ANALYSIS_DIR / "ner_predictions.json")
    predict.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    predict.add_argument("--context-mode", choices=CONTEXT_MODES)
    predict.add_argument("--threshold", type=float, default=0.0)
    tune = subparsers.add_parser("tune-threshold")
    tune.add_argument("--validation-file", type=Path, default=DEFAULT_DATA_DIR / "validation.jsonl")
    tune.add_argument("--prediction-file", type=Path, required=True)
    tune.add_argument("--output-file", type=Path, default=ANALYSIS_DIR / "equation_ner_threshold.json")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--test-file", type=Path, default=DEFAULT_DATA_DIR / "test.jsonl")
    evaluate.add_argument("--prediction", action="append", required=True, help="NAME=PATH; repeat for each model")
    evaluate.add_argument("--output-file", type=Path, default=ANALYSIS_DIR / "equation_ner_evaluation.json")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "export-review-template":
        counts = export_review_template(args.equations_file, args.existing_file, args.output_file)
        print(f"Exported {counts['total']} records: {counts['reviewed']} reviewed, {counts['needs_review']} need review")
    elif args.command == "build-data":
        report = build_dataset(args.analysis_file, args.output_dir, args.train_ratio, args.validation_ratio, args.seed, args.max_tokens)
        print(f"Usable examples: {report['usable_examples']}; unreviewed or invalid: {report['unreviewed_or_invalid_records']}")
    elif args.command == "train":
        train_model(args.data_dir, args.model_dir, args.base_model, args.epochs, args.batch_size, args.learning_rate, args.weight_decay, args.context_mode, args.seed, args.runs_dir)
        print(f"Saved model to {args.model_dir}")
    elif args.command == "predict":
        summary = predict_file(args.input_file, args.output_file, args.model_dir, args.context_mode, args.threshold)
        print(f"Predicted meanings for {summary['filled']} of {summary['total']}")
    elif args.command == "tune-threshold":
        report = tune_threshold(args.validation_file, args.prediction_file, args.output_file)
        print(f"Recommended validation threshold: {report['recommended_threshold']:.6f}")
    elif args.command == "evaluate":
        report = evaluate_files(args.test_file, args.prediction, args.output_file)
        print(f"Evaluated {len(report['models'])} models on {report['examples']} examples")


if __name__ == "__main__":
    main()
