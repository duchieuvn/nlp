# Equation NER Analysis Workflow

This folder contains the custom NER implementation described in
`notes/custom_ner_plan.md`.

The goal is to train a SciBERT token-classification model that extracts an
equation meaning span from the local text window around `[EQUATION]`.

## Inputs

- `analysis_meaning.json`: reviewed equation meanings and source windows.

Each record is expected to contain:

```json
{
  "meaning": "Markovian quantum master equation",
  "window": "our Markovian quantum master equation reads [EQUATION]"
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

The builder only keeps examples where the reviewed `meaning` can be found as a
contiguous token span in the source `window`. Unmatched examples are listed in
`dataset_report.json` for manual review.

## Train

```bash
python source/analysis/equation_ner.py train
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

## Current Dataset Snapshot

The current `analysis_meaning.json` build produces:

- 82 reviewed records.
- 77 usable BIO examples.
- 5 skipped examples that need review or rewritten spans.
- Split sizes: 61 train, 7 validation, 9 test.
