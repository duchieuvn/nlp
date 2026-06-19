from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import random
import re
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_FILE = ANALYSIS_DIR / "analysis_meaning.json"
DEFAULT_DATA_DIR = ANALYSIS_DIR / "ner_data"
DEFAULT_MODEL_DIR = ANALYSIS_DIR / "scibert-equation-ner-final"
DEFAULT_BASE_MODEL = "allenai/scibert_scivocab_uncased"
EQUATION_MARKER = "[EQUATION]"
DEFAULT_MAX_TOKENS = 256
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
        if not normalized:
            continue
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
    if len(tokens) <= max_tokens:
        return tokens, tags

    labeled_indices = [
        index
        for index, tag in enumerate(tags)
        if tag in {"B-EQ_NAME", "I-EQ_NAME"}
    ]
    marker_indices = [
        index
        for index, token in enumerate(tokens)
        if token == EQUATION_MARKER
    ]
    important_indices = labeled_indices + marker_indices
    if not important_indices:
        return tokens[:max_tokens], tags[:max_tokens]

    required_start = min(important_indices)
    required_end = max(important_indices) + 1
    required_length = required_end - required_start
    if required_length >= max_tokens:
        start = max(0, required_start)
        end = min(len(tokens), start + max_tokens)
        return tokens[start:end], tags[start:end]

    spare = max_tokens - required_length
    start = max(0, required_start - spare // 2)
    end = start + max_tokens
    if end > len(tokens):
        end = len(tokens)
        start = max(0, end - max_tokens)

    return tokens[start:end], tags[start:end]


def label_window_tokens(
    window: str,
    meaning: str,
    max_tokens: int,
) -> tuple[list[str], list[str]] | None:
    tokens = tokenize_text(window)
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


def iter_labeled_records(analysis_file: Path) -> list[dict[str, Any]]:
    data = json.loads(analysis_file.read_text(encoding="utf-8"))
    records = []

    for paper_id, equations in data.items():
        for equation_id, payload in equations.items():
            meaning = normalize_text(payload.get("meaning", ""))
            window = normalize_text(payload.get("window", ""))
            if not meaning or not window:
                continue
            records.append(
                {
                    "paper_id": paper_id,
                    "equation_id": equation_id,
                    "meaning": meaning,
                    "window": window,
                }
            )

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
            record["meaning"],
            max_tokens,
        )
        if labeled is None:
            skipped.append(
                {
                    "paper_id": record["paper_id"],
                    "equation_id": record["equation_id"],
                    "meaning": record["meaning"],
                    "reason": "meaning tokens were not found as a contiguous span in window",
                }
            )
            continue

        tokens, tags = labeled
        examples.append(
            NerExample(
                paper_id=record["paper_id"],
                equation_id=record["equation_id"],
                tokens=tokens,
                ner_tags=tags,
                meaning=record["meaning"],
                window=record["window"],
            )
        )

    return examples, skipped


def split_examples(
    examples: list[NerExample],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, list[NerExample]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def example_to_json(example: NerExample) -> dict[str, Any]:
    return {
        "paper_id": example.paper_id,
        "equation_id": example.equation_id,
        "tokens": example.tokens,
        "ner_tags": example.ner_tags,
        "meaning": example.meaning,
        "window": example.window,
    }


def write_jsonl(path: Path, examples: list[NerExample]) -> None:
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
    records = iter_labeled_records(analysis_file)
    examples, skipped = build_examples(records, max_tokens)
    splits = split_examples(examples, train_ratio, validation_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples_ in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_examples_)

    report = {
        "source_file": str(analysis_file),
        "output_dir": str(output_dir),
        "label_set": list(LABELS),
        "max_tokens": max_tokens,
        "total_records": len(records),
        "usable_examples": len(examples),
        "skipped_examples": len(skipped),
        "splits": {name: len(items) for name, items in splits.items()},
        "skipped": skipped,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class EquationNerDataset:
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tokenized = self.tokenizer(
            record["tokens"],
            truncation=True,
            is_split_into_words=True,
        )
        label_ids = []
        previous_word_index = None

        for word_index in tokenized.word_ids():
            if word_index is None:
                label_ids.append(-100)
            elif word_index != previous_word_index:
                label_ids.append(LABEL2ID[record["ner_tags"][word_index]])
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
        raise RuntimeError(
            "Training requires transformers, torch, and their dependencies. "
            "Install them before running the train command."
        ) from exc

    return (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )


def train_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: float = 5.0,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> None:
    (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    ) = load_training_dependencies()

    train_records = read_jsonl(data_dir / "train.jsonl")
    validation_records = read_jsonl(data_dir / "validation.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [EQUATION_MARKER]}
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = EquationNerDataset(train_records, tokenizer)
    validation_dataset = EquationNerDataset(validation_records, tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(model_dir.parent / "scibert-equation-ner-runs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))


def load_inference_pipeline(model_dir: Path) -> Any:
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference requires transformers. Install it before running predict."
        ) from exc

    return pipeline(
        "ner",
        model=str(model_dir),
        tokenizer=str(model_dir),
        aggregation_strategy="simple",
    )


def predict_meaning(window: str, ner_pipeline: Any) -> dict[str, Any]:
    entities = ner_pipeline(window)
    eq_name_entities = [
        entity
        for entity in entities
        if entity.get("entity_group") == "EQ_NAME"
    ]
    if not eq_name_entities:
        return {
            "meaning": "",
            "confidence": 0.0,
            "entities": entities,
        }

    best = max(eq_name_entities, key=lambda entity: entity.get("score", 0.0))
    return {
        "meaning": normalize_text(best.get("word", "")),
        "confidence": float(best.get("score", 0.0)),
        "entities": entities,
    }


def predict_file(
    input_file: Path = DEFAULT_ANALYSIS_FILE,
    output_file: Path = ANALYSIS_DIR / "ner_predictions.json",
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    ner_pipeline = load_inference_pipeline(model_dir)
    data = json.loads(input_file.read_text(encoding="utf-8"))
    predictions = {}
    total = 0
    filled = 0

    for paper_id, equations in data.items():
        predictions[paper_id] = {}
        for equation_id, payload in equations.items():
            total += 1
            result = predict_meaning(payload.get("window", ""), ner_pipeline)
            if result["meaning"]:
                filled += 1
            predictions[paper_id][equation_id] = {
                "gold_meaning": payload.get("meaning", ""),
                "predicted_meaning": result["meaning"],
                "confidence": result["confidence"],
            }

    summary = {
        "model_dir": str(model_dir),
        "input_file": str(input_file),
        "total": total,
        "filled": filled,
        "predictions": predictions,
    }
    output_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def add_build_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("build-data")
    parser.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.set_defaults(command="build-data")


def add_train_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.set_defaults(command="train")


def add_predict_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("predict")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ANALYSIS_DIR / "ner_predictions.json",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.set_defaults(command="predict")


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Build, train, and run the equation-meaning NER model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_build_parser(subparsers)
    add_train_parser(subparsers)
    add_predict_parser(subparsers)
    return parser


def main() -> None:
    args = make_parser().parse_args()

    if args.command == "build-data":
        report = build_dataset(
            analysis_file=args.analysis_file,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
        print(f"Total records: {report['total_records']}")
        print(f"Usable examples: {report['usable_examples']}")
        print(f"Skipped examples: {report['skipped_examples']}")
        print(f"Wrote dataset to {report['output_dir']}")
    elif args.command == "train":
        train_model(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print(f"Saved model to {args.model_dir}")
    elif args.command == "predict":
        summary = predict_file(
            input_file=args.input_file,
            output_file=args.output_file,
            model_dir=args.model_dir,
        )
        print(f"Predicted meanings for {summary['filled']} of {summary['total']}")
        print(f"Wrote predictions to {args.output_file}")


if __name__ == "__main__":
    main()
