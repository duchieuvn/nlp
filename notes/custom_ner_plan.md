# One-Click MathBERT Baseline

Open `source/s4b.py` and press **Run Python File** in the IDE.

The script automatically:

1. Installs `torch` and `transformers` if they are missing.
2. Downloads `witiko/mathberta` on the first run.
3. Reads `data/3_equations.json` directly.
4. Uses MathBERTA embeddings to rank literal equation-name candidates.
5. Writes `data/4b_mathbert_baseline.json`.

It does not use the 82 reviewed examples, a training dataset, SciBERT, or a QA
model. MathBERTA is a base masked-language model rather than a trained QA/NER
model, so this is an honest zero-label embedding baseline.
