# Equation NER Analysis Workflow

This folder contains the custom NER implementation described in
`notes/custom_ner_plan.md`.

The goal is to train a SciBERT token-classification model that extracts an
equation meaning span from the local text window around `[EQUATION]`.

## Inputs

- `analysis_meaning.json`: reviewed equation meanings and source windows.

Each record contains an explicit review state and exact source offsets:

```json
{
  "meaning": "Markovian quantum master equation",
  "window": "our Markovian quantum master equation reads [EQUATION]",
  "equation": "\\partial_t \\rho = ...",
  "has_answer": true,
  "start": 4,
  "end": 38,
  "review_status": "reviewed"
}
```

## Build BIO Data

```bash
python source/analysis/equation_ner.py build-data
```

This writes:

- `source/analysis/ner_data/train.jsonl`
- `source/analysis/ner_data/validation.jsonl`
- `source/analysis/ner_data/test.jsonl`
- `source/analysis/ner_data/dataset_report.json`

The builder includes reviewed positive and no-answer examples. Unreviewed or
invalid records are listed in `dataset_report.json` and excluded. Dataset
splits are grouped by paper.

## Train

```bash
python source/analysis/equation_ner.py train --context-mode marker
```

By default, this fine-tunes:

```text
allenai/scibert_scivocab_uncased
```

and saves the final model to:

```text
source/analysis/scibert-equation-ner-final
```

Training requires `transformers`, `torch`, and the SciBERT model files.

## Predict

```bash
python source/analysis/equation_ner.py predict
```

This loads the trained model and writes predictions to:

```text
source/analysis/ner_predictions.json
```

## Label Set

- `O`: outside an equation-name span.
- `B-EQ_NAME`: first token of an equation-name span.
- `I-EQ_NAME`: continuation token inside an equation-name span.

Use `--base-model witiko/mathberta --context-mode formula` for the math-aware
experiment. See `notes/mathbert_extraction_plan.md` for the comparison and
promotion criteria.
