# Equation Multi-Vector Embedding Pipeline

**Target document:** `source2/plan/equation_embedding_pipeline_plan.md`

## Summary

Build the pipeline through creation of one paper-local embedding space containing multiple vectors per reviewed equation:

```text
Download HTML
→ Build raw documents
→ Build raw retrieval chunks
→ Extract reviewed equations
→ Build multi-vector MathBERT spaces
```

`data/3_equations.json` remains authoritative. Later equation-meaning, symbol-meaning, and relation retrieval will all use the same paper embedding space.

## Flat Layout

```text
source2/pipeline/
  config.py
  run_embedding_pipeline.py
  stage01_download_html.py
  stage02_build_documents.py
  stage03_build_chunks.py
  stage04_extract_equations.py
  stage05_build_embeddings.py
```

Each stage is one self-contained file, imports only `config.py`, exposes `run() -> dict`, validates its inputs, and atomically replaces its outputs.

## Stage 1: Download HTML

`stage01_download_html.py`:

- Read reviewed paper IDs in paper-list order.
- Reuse structurally valid cached HTML.
- Download missing files from arXiv with bounded concurrency.
- Retry timeouts, 429, and 5xx responses with backoff.
- Validate HTML before atomic storage.
- Record URL, attempts, status, size, checksum, and failure reason.

Outputs:

```text
data/html/<paper_id>.html
data/pipeline/download_report.json
```

## Stage 2: Build Raw Documents

`stage02_build_documents.py` parses successful HTML without selecting reviewed equations yet.

Preserve:

- Title and section hierarchy.
- Paragraphs and sentences with stable IDs and offsets.
- Every display-equation DOM block.
- Equation anchor, visible label, MathML, and original LaTeX.
- Document order and explicit equation references.
- Nearest prose sentences around every raw equation.

Output:

```text
data/pipeline/documents/<paper_id>.json
```

Validate unique IDs, parent links, sentence offsets, document order, and cross-reference sources.

## Stage 3: Build Retrieval Chunks

`stage03_build_chunks.py` creates:

```text
sentence
paragraph
section_context
raw_equation
raw_equation_neighborhood
cross_reference
```

Every chunk contains its source IDs, raw equation anchor, document order, section metadata, text, and visible equation labels.

Raw equation neighborhoods contain up to five prose sentences before and after the equation.

Output:

```text
data/pipeline/chunks/<paper_id>.json
```

At this point equations still use raw DOM-based IDs.

## Stage 4: Extract Reviewed Equations

`stage04_extract_equations.py` matches reviewed equations to raw HTML equations in this order:

1. Annotation DOM ID.
2. Reviewed audit anchor.
3. Unique visible label.
4. Visible label plus normalized LaTeX.
5. Exact normalized LaTeX.
6. Unique high-overlap normalized LaTeX.

Ambiguous matches are rejected.

Each output equation contains:

```json
{
  "paper_id": "...",
  "equation_id": "1",
  "latex": "...",
  "raw_equation_id": "...",
  "anchor_id": "...",
  "section_id": "...",
  "document_order": 0,
  "match_method": "...",
  "before_sentences": [],
  "after_sentences": [],
  "audit": {}
}
```

The before and after arrays contain at most five closest prose sentences on each side.

Outputs:

```text
data/pipeline/equations/<paper_id>.json
data/pipeline/equation_alignment_report.json
```

Every reviewed equation in a successful HTML paper must resolve uniquely.

## Stage 5: Build Multi-Vector Spaces

`stage05_build_embeddings.py` loads `witiko/mathberta` with `AutoTokenizer` and `AutoModel`.

Each equation contributes up to twelve vectors:

| Vector kind | Count | Embedded content |
| --- | ---: | --- |
| `equation` | 1 | Equation LaTeX only |
| `summary` | 1 | Five preceding sentences, equation, five following sentences |
| `before_sentence` | 0–5 | Equation LaTeX plus one preceding sentence |
| `after_sentence` | 0–5 | Equation LaTeX plus one following sentence |

All vector text is derived only from the equation and its surrounding prose.

### Input Representations

Equation vector:

```text
equation: <latex>
```

Summary vector:

```text
before: <up to five sentences>
equation: <latex>
after: <up to five sentences>
```

Sentence vector:

```text
equation: <latex>
context: <one exact surrounding sentence>
position: before|after
distance: <distance from equation>
```

Position and distance are structural markers, not generated semantic text.

### Token Budget

Use a maximum of 512 tokens including special tokens.

For `summary` vectors:

- Reserve up to 256 tokens for the equation.
- Split remaining tokens between before and after context.
- Keep the closest before-context tail and after-context head.
- Redistribute unused tokens from one side to the other.
- For an equation exceeding its budget, retain equal head and tail portions.

For sentence-conditioned vectors:

- Reserve up to 320 tokens for the equation.
- Use the remaining budget for the complete context sentence.
- Truncate the equation before truncating a short context sentence.
- Record every truncation decision.

For equation-only vectors, use the complete equation when possible; otherwise retain equal head and tail portions.

### Embedding Calculation

For every representation:

1. Tokenize deterministically.
2. Run MathBERT in evaluation and inference mode.
3. Mean-pool the final hidden state with the attention mask.
4. L2-normalize the result.
5. Store it as `float32`.

CUDA is used when available, with automatic CPU fallback.

## Persisted Paper Space

Each paper produces:

```text
data/pipeline/embeddings/<paper_id>.npz
data/pipeline/embeddings/<paper_id>.json
```

The `.npz` file contains:

```text
embeddings: float32[number_of_vectors, 768]
```

The metadata file contains:

```json
{
  "paper_id": "...",
  "model": "witiko/mathberta",
  "pooling": "attention_mask_mean",
  "normalized": true,
  "dimension": 768,
  "rows": [
    {
      "row": 0,
      "vector_id": "paper:equation:1:summary",
      "equation_id": "1",
      "vector_kind": "summary",
      "sentence_id": null,
      "context_position": null,
      "context_distance": 0,
      "source_text": "...",
      "input_token_count": 0,
      "equation_truncated": false,
      "context_truncated": false,
      "input_sha256": "..."
    }
  ]
}
```

Stable vector ordering is:

1. Equation ID order.
2. `equation`.
3. `summary`.
4. Before sentences from nearest to farthest.
5. After sentences from nearest to farthest.

## Retrieval Interface

`stage05_build_embeddings.py` also exposes:

```python
load_paper_space(paper_id)
embed_query(query_text)
search_paper(
    paper_id,
    query_text,
    top_k,
    vector_kinds=None,
    exclude_equation_ids=None,
    group_by_equation=False,
)
```

Search uses exact cosine similarity through matrix-vector multiplication.

When `group_by_equation=True`, retain the highest-scoring vector per equation and return the winning vector’s kind and source sentence.

Recommended downstream usage:

- Equation meaning: search `summary`, `before_sentence`, and `after_sentence`.
- Symbol meaning: search `before_sentence` and `after_sentence`.
- Relations: compare `summary` vectors primarily and `equation` vectors as mathematical-similarity evidence.

## Orchestration

`run_embedding_pipeline.py` runs stages 1–5 sequentially without arguments.

Valid HTML remains cached. All derived artifacts are rebuilt on each run through temporary directories and atomic replacement.

The build report records:

- Download successes and failures.
- Raw and reviewed equation counts.
- Alignment methods and unresolved equations.
- Vectors per equation and paper.
- Truncation counts.
- Model, device, dimension, runtime, and output checksums.

Output:

```text
data/pipeline/build_report.json
```

## Test Plan

- Test cached, corrupt, retryable, and failed HTML downloads.
- Verify raw equation discovery, stable document order, and cross-references.
- Validate every chunk’s source IDs.
- Test every equation matching strategy and ambiguity rejection.
- Verify a maximum of five context sentences on each side.
- Verify vector counts: `2 + before_count + after_count`.
- Confirm sentence vectors preserve exact source sentences.
- Test token budgeting and head/tail truncation.
- Confirm finite, normalized `float32[768]` vectors.
- Verify stable row ordering and metadata hashes.
- Confirm paper-local isolation and exact cosine ordering.
- Test vector-kind filters, equation exclusion, and grouped results.
- Require deterministic outputs across repeated runs.

## Assumptions

- Only reviewed target equations enter the embedding spaces.
- Each equation contributes an equation vector, summary vector, and available sentence vectors.
- No paragraph or unrelated chunk vectors enter this space.
- MathBERT is inference-only and non-generative.
- All downstream semantic retrieval uses this single paper-local space.
- Extraction of meanings, symbols, and relations begins only after this plan’s outputs are complete.
