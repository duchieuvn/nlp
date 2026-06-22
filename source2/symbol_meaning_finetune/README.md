# Symbol Meaning MathBERT Fine-Tuning

This folder contains a standalone, one-command pipeline for fine-tuning a
MathBERT cross-encoder to classify symbol-to-phrase relations.

## Run

From this directory:

```bash
python run.py
```

No command-line arguments are required. Paths and hyperparameters are defined
in `config.py`.

The runner performs the complete workflow:

1. Builds weak relation labels from accepted regex definitions and rejected
   same-paper BM25 evidence.
2. Splits examples by paper to prevent context leakage.
3. Fine-tunes `witiko/mathberta` with class-balanced sampling.
4. Calibrates definition acceptance on validation papers.
5. Evaluates on held-out test papers.
6. Writes the checkpoint, inference thresholds, JSON metrics, and Markdown
   performance report.

## Outputs

- `data/source2/symbol_meaning_finetune/checkpoint/`
- `data/source2/symbol_meaning_finetune/metrics.json`
- `data/source2/symbol_meaning_finetune/performance_report.md`
- `data/source2/symbol_meaning_finetune/dataset_summary.json`

The reported metrics use held-out weak labels. A manually reviewed benchmark
is still required to establish real-world precision before promoting model
predictions into the production symbol-meaning dataset.
