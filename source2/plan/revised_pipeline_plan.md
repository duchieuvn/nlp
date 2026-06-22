# Revised Flat End-to-End Equation Pipeline

## Summary

Reimplement the complete pipeline in one flat, non-package directory `source2/pipeline/`. Each processing stage is exactly one Python file, with one shared `config.py` and one `run_pipeline.py` orchestrator.

`data/3_equations.json` remains the authoritative reviewed equation corpus. The final output preserves the current schema:

```json
{
  "paper_id": {
    "equation_id": {
      "equation": "...",
      "meaning": "...",
      "symbols": { "canonical": "definition" },
      "relations": { "target_id": { "grade": "...", "description": "..." } }
    }
  }
}
```

Audit metadata stays in intermediate files and is excluded from `data/final_data.json`.

## Key Architectural Change

After building retrieval chunks (stage 3), the pipeline extracts equations and builds a **single dense embedding space per paper** from the equation itself and its surrounding text. This shared embedding index is then used by all downstream retrieval tasks — equation meaning, symbol meanings, and relations — rather than rebuilding separate indices for each task.

## Flat Layout

```text
source2/pipeline/
  config.py
  run_pipeline.py
  stage01_download_html.py
  stage02_build_documents.py
  stage03_build_chunks.py
  stage04_extract_equations.py
  stage05_build_embedding_space.py
  stage06_extract_meanings.py
  stage07_extract_symbol_meanings.py
  stage08_build_relations.py
  stage09_postprocess_meanings.py
  stage10_export_final.py
```

Each stage exposes `run() -> dict`, reads only `config.py` and prior JSON artifacts, validates its inputs, atomically replaces its output directory, and returns counts, timing, warnings, and failures.

## Shared Configuration

`config.py` contains every path and operational setting:

- Paper list, reviewed equations, annotations, HTML cache, intermediate root, and final output.
- arXiv URL, user agent, timeout, retries, backoff, concurrency, and response validation.
- Chunk types, sentence segmentation model, and top-k retrieval values.
- Embedding model name, device policy, batch size, and index persistence path.
- Equation MathBERT model, checkpoint/cache path, device policy, and batch size.
- Meaning, symbol-definition, and relation thresholds.
- Required final fields, valid relation grades, deterministic seed, and logging paths.

No stage accepts command-line arguments. Configuration changes happen only in this file.

## Pipeline Stages

### 1. Download HTML — `stage01_download_html.py`

Read paper IDs from the reviewed equation corpus in paper-list order. Reuse cached HTML only when it is non-empty and contains a valid HTML document. Download missing or invalid files from arXiv with bounded concurrency, retries for timeouts/429/5xx responses, exponential backoff, and atomic writes. Record URL, status, attempts, size, checksum, and error. Failed papers are skipped downstream and later exported as empty paper objects.

### 2. Build Structured Documents — `stage02_build_documents.py`

Parse successful HTML with BeautifulSoup and `lxml`. Preserve title, section hierarchy, paragraphs, sentence text and offsets, equation DOM blocks, MathML/LaTeX, visible labels, and explicit equation references. Align reviewed equations using annotation DOM ID, audit anchor, visible label plus LaTeX, then exact normalized LaTeX. Fail the stage if a reviewed equation in otherwise valid HTML cannot be uniquely aligned. Write one document per non-empty successful paper.

### 3. Build Retrieval Chunks — `stage03_build_chunks.py`

Produce sentence, paragraph, equation-neighborhood, section-aware, and cross-reference chunks. Every chunk carries stable source IDs, section metadata, nearby equation IDs, text, document order, and source type. Deduplicate equivalent text while preferring sentence-level provenance. Validate unique chunk IDs and valid source references.

### 4. Extract Equations — `stage04_extract_equations.py`

Using the aligned equation blocks from stage 2, extract the canonical form of each reviewed equation along with its local context window. For each equation, collect:

- The equation text and LaTeX/MathML representation.
- The `N` sentences immediately before and after the equation in document order (configurable window size, default 3 sentences each side).
- The containing paragraph text.
- Section title and path.
- All visible equation labels and cross-reference anchors.
- Symbols present in the equation (from MathML composite reconstruction; LaTeX tokenizer fallback when MathML is unavailable).

Output is one JSON file per paper: a map from `equation_id` to its extracted record. This stage makes no LLM or embedding calls — it is purely structural extraction from stage 2 and 3 artifacts.

### 5. Build Embedding Space — `stage05_build_embedding_space.py`

Build one dense vector index per paper. Each index entry corresponds to a chunk from stage 3 that is directly associated with at least one reviewed equation — specifically sentence, paragraph, and equation-neighborhood chunks within the context window of each equation.

**Index construction:**

- For each paper, collect all chunks whose `nearby_equation_ids` overlap with any reviewed equation ID for that paper.
- Encode each chunk text with a sentence-transformer model (configured in `config.py`; default `sentence-transformers/all-MiniLM-L6-v2` or a domain-specific model).
- Store the resulting FAISS flat-L2 index alongside a metadata sidecar that maps vector row index → `{ chunk_id, equation_ids, section, source_type, text }`.
- Persist both the index and sidecar to `data/pipeline/embeddings/<paper_id>/`.

**Index is shared across all downstream stages.** No stage rebuilds it. Each downstream stage loads the index once per paper, issues its queries, and retrieves the top-k nearest chunks.

**Query protocol (used in stages 6, 7, 8):**

```
embed(query_text)  →  FAISS.search(k=top_k)  →  ranked chunk list with scores
```

Queries are deterministic strings assembled from structured fields (equation text, symbol name, section, etc.) — not generated text.

### 6. Extract Equation Meanings — `stage06_extract_meanings.py`

For each reviewed equation, assemble a meaning query from equation ID, section, important symbols, and meaning cues. Retrieve top-k chunks from the paper's embedding index (stage 5). Combine embedding scores with BM25 scores computed over the retrieved set, proximity to the equation, section match, and cue-pattern features.

Apply extractive candidate generation from the retrieved chunks: explicit equation references, before-equation descriptive clauses, named-equation constructions, and deterministic cue patterns. Hard-filter procedural, incomplete, symbol-definition, and math-heavy spans. Use `witiko/mathberta` only as a non-generative reranker. Emit an empty meaning when no candidate passes the configured score threshold.

### 7. Extract Symbol Meanings — `stage07_extract_symbol_meanings.py`

For each symbol in each reviewed equation, assemble queries using the exact LaTeX form, canonical aliases, nearby equation ID, and definition cues. Retrieve top-k chunks from the paper's embedding index (stage 5). Apply ordered high-precision rules over retrieved text: `where X is`, `X denotes`, `let X be`, reverse definitions, noun-before-symbol constructions, coordinated clauses, and cardinality-matched `respectively` expressions.

Require exact composite-symbol evidence for decorated symbols. Reject appositive-only, vague, dangling, stop-word-only, artifact-containing, math-heavy, self-referential, and overlong spans. Keep definitions verbatim and record character offsets, pattern, confidence, evidence IDs, and explicit rejection reasons.

### 8. Build Relations — `stage08_build_relations.py`

For each directed equation pair within a paper, assemble a relation query combining both equation texts and key symbols. Retrieve top-k chunks from the paper's embedding index (stage 5). Score explicit citations, derivation, equivalence, special-case cues, shared canonical symbols, section proximity, and context similarity over the retrieved evidence. Map fixed thresholds to `strong`, `potential`, or `none`. Use only the fixed descriptions `explicit citation`, `derived from`, `equivalent`, `special case`, `shares symbols`, and `same section context`. Guarantee complete relation coverage to every other equation.

### 9. Postprocess Meanings — `stage09_postprocess_meanings.py`

Preserve raw extraction separately. Mechanically remove trailing rephrasing artifacts, causal reporting wrappers, procedural fragments, and symbol-definition clauses. Reduce retained meanings to complete extractive noun phrases of at most 12 natural-language words. Never paraphrase or generate text. Store original meaning, processed meaning, transformation rule, removed span, and source provenance.

### 10. Export Final Data — `stage10_export_final.py`

Join reviewed equations, postprocessed meanings, symbol definitions, and relations by paper and equation ID. Validate equation text equality, component coverage, canonical symbol uniqueness, relation completeness, grades, descriptions, and strict field order. Preserve reviewed paper-list order. Papers with no reviewed equations or failed HTML become `{}`. Write `data/final_data.json` atomically only after all validation succeeds.

## Orchestration and Artifacts

`run_pipeline.py` invokes all stages sequentially with no arguments. HTML remains a validated download cache; every derived stage is always rebuilt into a temporary directory and swapped into place only after validation. A failure stops downstream execution while preserving the last complete outputs.

Intermediate data is written under:

```text
data/pipeline/documents/
data/pipeline/chunks/
data/pipeline/equations/
data/pipeline/embeddings/
data/pipeline/meanings_raw/
data/pipeline/symbol_meanings/
data/pipeline/relations/
data/pipeline/meanings/
data/pipeline/reports/
```

The runner writes a consolidated build report containing input fingerprints, successful and skipped papers, unresolved equations, stage counts, runtimes, model/device information, rejection totals, and final coverage.

## Data Flow Summary

```
stage01 → HTML cache
stage02 → documents/          (structured document per paper)
stage03 → chunks/             (sentence, paragraph, neighborhood chunks)
stage04 → equations/          (equation + context window per paper)
stage05 → embeddings/         (FAISS index + metadata sidecar per paper)
             ↓           ↓           ↓
stage06      stage07     stage08     (all query the same index)
meanings_raw/ symbol_meanings/ relations/
             ↓
stage09 → meanings/
             ↓
stage10 → data/final_data.json
```

## Test Plan

- Mock arXiv success, cache reuse, corrupt HTML, retryable errors, and permanently missing HTML.
- Verify equation alignment precedence, ambiguity rejection, section hierarchy, sentence offsets, and cross-references.
- Test MathML composite-symbol reconstruction and LaTeX fallback for Greek, subscripts, superscripts, primes, and decorators.
- Verify equation context window boundaries, correct N-sentence capture, and correct symbol extraction.
- Verify embedding index construction: correct chunk filtering, vector count, and metadata sidecar integrity.
- Verify deterministic query assembly and correct top-k retrieval from the shared index for all three downstream tasks.
- Test meaning hard filters, MathBERT reranking, CPU fallback, and extractive-only output.
- Test symbol definition patterns, coordination, `respectively`, single-letter alias protection, and rejection reasons.
- Verify complete directed relation matrices and fixed grade/description invariants.
- Run an end-to-end fixture with one valid paper, one empty reviewed paper, and one failed-HTML paper.
- Assert deterministic paper/equation ordering and exact compatibility with the four-field final schema.
- Acceptance requires atomic output, zero unresolved aligned equations for successful papers, no unsupported non-empty values, and complete relations.

## Assumptions

- `data/3_equations.json` and annotations are reviewed inputs, not regenerated by this pipeline.
- Only successfully downloaded HTML papers are processed; failed papers remain as empty objects in the final export.
- Cached HTML is not redownloaded when valid; all derived artifacts are always rebuilt.
- The embedding index is built once per run and shared by stages 6, 7, and 8 — no stage rebuilds it.
- Equation MathBERT is inference-only and non-generative.
- Symbol extraction and symbol meanings are deterministic and rule-based.
- Existing source2 directories remain untouched until the flat pipeline reproduces and validates `final_data.json`.
