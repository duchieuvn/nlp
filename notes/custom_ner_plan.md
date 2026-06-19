# Custom NER Plan for Equation Meaning Extraction

## Goal

Transition the equation meaning extractor from heuristic rules to a custom Named Entity Recognition (NER) pipeline. The target model should learn the language patterns around `[EQUATION]` markers and identify equation names or concepts directly from surrounding text.

This approach replaces manually maintained grammatical rules with token-level sequence labeling using SciBERT and the Hugging Face `transformers` library.

## Why Use Custom NER

The current heuristic script depends on regex patterns, dependency rules, and hand-built grammatical cues. A custom NER model can learn from successful examples and generalize to phrasing that the rules miss.

Expected benefits:

- Reduce reliance on brittle regex patterns.
- Learn phrase shapes around scientific equations.
- Use local context around `[EQUATION]` as model input.
- Produce a cleaner production extractor after training.
- Keep the extraction pipeline programmatic and auditable.

## Step 1: Build a BIO Dataset

Sequence labeling requires one label per token. Instead of storing only character offsets, each training example should be converted into tokens and BIO tags.

Example source sentence:

```text
our master equation reads [EQUATION]
```

Example training record:

```json
{
  "tokens": ["our", "master", "equation", "reads", "[EQUATION]"],
  "ner_tags": ["O", "B-EQ_NAME", "I-EQ_NAME", "O", "O"]
}
```

Label meanings:

- `B-EQ_NAME`: beginning of an equation name or concept.
- `I-EQ_NAME`: continuation of an equation name or concept.
- `O`: token outside the target entity.

Initial data source:

- Start from the successfully extracted examples in the current rule-based output.
- Convert each confirmed equation meaning into BIO labels.
- Store the result as JSONL or CSV for training.

## Step 2: Tokenize and Align Labels

SciBERT uses subword tokenization, so one original token may become multiple model tokens. The BIO labels must be aligned with the tokenizer output before training.

Use `allenai/scibert_scivocab_uncased` and register `[EQUATION]` as an additional special token.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer.add_special_tokens({"additional_special_tokens": ["[EQUATION]"]})

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

Implementation notes:

- Use `-100` for special tokens and ignored subword continuations.
- Keep labels on the first subword only.
- Confirm that each tokenized example has the same number of model tokens and aligned label IDs.

## Step 3: Initialize the Token Classification Model

Load SciBERT with a token classification head. The classification head maps each token representation to one of the BIO labels.

```python
from transformers import AutoModelForTokenClassification

label2id = {
    "O": 0,
    "B-EQ_NAME": 1,
    "I-EQ_NAME": 2,
}
id2label = {
    0: "O",
    1: "B-EQ_NAME",
    2: "I-EQ_NAME",
}

model = AutoModelForTokenClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

model.resize_token_embeddings(len(tokenizer))
```

Important detail:

- Resizing token embeddings is required because `[EQUATION]` was added to the tokenizer.

## Step 4: Fine-Tune the Model

Use the Hugging Face `Trainer` API to fine-tune SciBERT on the BIO-labeled equation context data.

```python
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./scibert-equation-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_validation_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./scibert-equation-ner-final")
```

Training expectations:

- The model should learn contextual triggers such as "reads", "is given by", "known as", "yields", and "form".
- Validation data should include examples not used during training.
- Evaluation should focus on exact phrase extraction, not just token-level accuracy.

## Step 5: Replace Heuristic Inference

After training, replace the large heuristic extraction block with a compact NER inference path.

```python
from transformers import pipeline

ner_pipeline = pipeline(
    "ner",
    model="./scibert-equation-ner-final",
    aggregation_strategy="simple",
)

window = (
    "tracing out the bath degrees of freedom yields the "
    "Markovian quantum master equation [EQUATION]"
)

entities = ner_pipeline(window)

for entity in entities:
    if entity["entity_group"] == "EQ_NAME":
        print(f"Extracted Concept: {entity['word']}")
        print(f"Confidence: {entity['score']:.4f}")
```

Expected output:

```text
Extracted Concept: Markovian quantum master equation
```

## Data Labeling Challenge

The success of this approach depends heavily on the training data. If the heuristic script only finds a small number of high-confidence examples, the remaining missed examples need a labeling strategy.

Key question:

- How should the missed examples be labeled so the model sees enough varied equation-name patterns?

Possible approaches:

- Manually review and label missed examples in `analysis_meaning.json`.
- Use the current heuristic output as weak labels, then correct them.
- Prioritize examples where the surrounding text includes clear cue verbs.
- Add a small annotation tool or script to speed up BIO labeling.
- Split reviewed examples into train, validation, and test sets.

## Suggested Implementation Stages

1. Export confirmed equation meaning examples into token/BIO format.
2. Create a review workflow for missed or low-confidence examples.
3. Build tokenizer alignment and dataset loading code.
4. Fine-tune SciBERT with a token classification head.
5. Evaluate phrase-level precision and recall.
6. Save the trained model and tokenizer.
7. Replace heuristic inference in `step4_eqn_meaning.py`.
8. Add audit trail fields for model confidence and extracted span.
9. Compare outputs against the current rule-based baseline.

## Audit Trail Fields

For each extracted meaning, store enough evidence to debug and justify the result.

Suggested audit payload:

```json
{
  "meaning_extraction": {
    "method": "SciBERT token classification",
    "model_path": "./scibert-equation-ner-final",
    "entity_group": "EQ_NAME",
    "confidence": 0.964,
    "source_text_slice": "our master equation reads [EQUATION]",
    "extracted_span": "master equation"
  }
}
```

## Success Criteria

- The model extracts compact equation names from local context.
- The output improves recall over the current heuristic extractor.
- Extracted spans remain evidence-based and traceable to source text.
- The production inference code is simpler than the current rule stack.
- The pipeline still records enough audit metadata for review.
