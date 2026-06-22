# Hybrid Symbol Meaning Extraction

This package preserves the `source2/symbol_meaning` JSON record schema while
adding a fine-tuned MathBERT cross-encoder for unresolved symbols.

## Workflow

1. Fine-tune or refresh the checkpoint:

   ```bash
   python entry_finetune.py
   ```

2. Extract regex-first, checkpoint-backed symbol meanings:

   ```bash
   python entry_extract.py
   ```

3. Create a balanced review sample:

   ```bash
   python entry_build_review.py
   ```

4. Fill `gold_relation` in `analysis/symbol_meaning2_review.json`, then
   calibrate against the reviewed labels:

   ```bash
   python entry_calibrate.py
   ```

5. Run `entry_extract.py` again with the reviewed threshold.

All paths, thresholds, and model settings live in `symbol_config.py`; the entry
points require no command-line arguments.

## Acceptance Rules

- High-confidence regex definitions always take precedence.
- Neural extraction runs only for unresolved symbols.
- `DEFINES_COMPLETE_SYMBOL` may populate `definition`.
- `DEFINES_BASE` may populate it only for symbols without semantic modifiers.
- Subscript and superscript qualifications remain in `audit`.
- Predictions below the calibrated probability or competing-label margin are
  rejected.
- Every neural definition is an extractive phrase from the evidence sentence.
