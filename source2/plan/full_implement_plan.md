# Flat End-to-End Equation Pipeline

## Summary

Reimplement the complete pipeline in one flat, non-package directory such as `source2/pipeline/`. Each processing stage is exactly one Python file, with one shared `config.py` and one `run_pipeline.py` orchestrator.

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

## Flat Layout

```text
source2/pipeline/
  config.py
  run_pipeline.py
  stage01_download_html.py
  stage02_build_documents.py
  stage03_build_chunks.py
  stage04_extract_symbols.py
  stage05_retrieve_evidence.py
  stage06_extract_meanings.py
  stage07_extract_symbol_meanings.py
  stage08_build_relations.py
  stage09_postprocess_meanings.py
  stage10_export_final.py
```

Each stage exposes `run() -> dict`, reads only `config.py` and prior JSON artifacts, validates its inputs, atomically replaces its output directory, and returns counts, timing, warnings, and failures. No shared models, utility, or package files are introduced; small JSON and validation helpers are intentionally local to each stage.

## Shared Configuration

`config.py` contains every path and operational setting:

- Paper list, reviewed equations, annotations, HTML cache, intermediate root, and final output.
- arXiv URL, user agent, timeout, retries, backoff, concurrency, and response validation.
- Chunk types, sentence segmentation model, retrieval tokenization, and top-k values.
- Equation MathBERT model, checkpoint/cache path, device policy, and batch size.
- Meaning, symbol-definition, and relation thresholds.
- Required final fields, valid relation grades, deterministic seed, and logging paths.
- Symbol cross-encoder remains disabled in the production pipeline.

No stage accepts command-line arguments. Configuration changes happen only in this file.

## Pipeline Stages

1. **Download HTML — `stage01_download_html.py`**

   Read paper IDs from the reviewed equation corpus in paper-list order. Reuse cached HTML only when it is non-empty and contains a valid HTML document. Download missing or invalid files from arXiv with bounded concurrency, retries for timeouts/429/5xx responses, exponential backoff, and atomic writes. Record URL, status, attempts, size, checksum, and error. Failed papers are skipped downstream and later exported as empty paper objects.

2. **Build Structured Documents — `stage02_build_documents.py`**

   Parse successful HTML with BeautifulSoup and `lxml`. Preserve title, section hierarchy, paragraphs, sentence text and offsets, equation DOM blocks, MathML/LaTeX, visible labels, and explicit equation references. Align reviewed equations using annotation DOM ID, audit anchor, visible label plus LaTeX, then exact normalized LaTeX. Fail the stage if a reviewed equation in otherwise valid HTML cannot be uniquely aligned. Write one document per non-empty successful paper.

3. **Build Retrieval Chunks — `stage03_build_chunks.py`**

   Produce sentence, paragraph, equation-neighborhood, section-aware, and cross-reference chunks. Every chunk carries stable source IDs, section metadata, nearby equation IDs, text, document order, and source type. Deduplicate equivalent text while preferring sentence-level provenance. Validate unique chunk IDs and valid source references.

4. **Extract Symbols — `stage04_extract_symbols.py`**

   Use the aligned equation MathML tree as the primary source. Reconstruct composite identifiers through `msub`, `msup`, `msubsup`, `mover`, `munder`, and decorator nodes rather than collecting flat `<mi>` values. Exclude operators, numbers, structural commands, and likely index-only variables. Preserve canonical name, original LaTeX forms, Unicode/Greek aliases, base, modifiers, and decorators. Fall back to the existing LaTeX tokenizer when usable MathML is unavailable.

5. **Retrieve Evidence — `stage05_retrieve_evidence.py`**

   Build one same-paper BM25 index per paper and reuse it for every query. Generate equation-meaning queries from equation ID, section, important symbols, and meaning cues. Generate symbol queries from exact LaTeX, canonical aliases, nearby equation, and definition cues. Retrieve from sentence, paragraph, and equation-neighborhood chunks, deduplicate results, and persist ranked evidence with scores and complete source provenance. This single stage supplies both extraction stages and avoids rebuilding BM25.

6. **Extract Equation Meanings — `stage06_extract_meanings.py`**

   Generate extractive candidates using explicit equation references, before-equation descriptive clauses, named-equation constructions, and deterministic cue patterns. Apply hard filters for procedural, incomplete, symbol-definition, and math-heavy spans. Use `witiko/mathberta` only as a non-generative reranker combined with BM25, proximity, section, citation, and cue features. Automatically fall back to CPU if CUDA is unavailable. Emit an empty meaning when no candidate passes the configured score.

7. **Extract Symbol Meanings — `stage07_extract_symbol_meanings.py`**

   Apply ordered high-precision rules for `where X is`, `X denotes`, `let X be`, reverse definitions, noun-before-symbol constructions, coordinated clauses, and cardinality-matched `respectively` expressions. Require exact composite-symbol evidence for decorated symbols. Reject appositive-only, vague, dangling, stop-word-only, artifact-containing, math-heavy, self-referential, and overlong spans. Keep definitions verbatim and record character offsets, pattern, confidence, evidence IDs, and explicit rejection reasons. Neural symbol inference remains disabled.

8. **Build Relations — `stage08_build_relations.py`**

   Create every directed equation pair within each successfully processed paper. Score explicit citations, derivation, equivalence, special-case cues, shared canonical symbols, section proximity, and context similarity. Map fixed thresholds to `strong`, `potential`, or `none`. Use only the fixed descriptions `explicit citation`, `derived from`, `equivalent`, `special case`, `shares symbols`, and `same section context`. Guarantee complete relation coverage to every other equation.

9. **Postprocess Meanings — `stage09_postprocess_meanings.py`**

   Preserve raw extraction separately. Mechanically remove trailing rephrasing artifacts, causal reporting wrappers, procedural fragments, and symbol-definition clauses. Reduce retained meanings to complete extractive noun phrases of at most 12 natural-language words. Never paraphrase or generate text. Store original meaning, processed meaning, transformation rule, removed span, and source provenance.

10. **Export Final Data — `stage10_export_final.py`**

    Join reviewed equations, postprocessed meanings, symbol definitions, and relations by paper and equation ID. Validate equation text equality, component coverage, canonical symbol uniqueness, relation completeness, grades, descriptions, and strict field order. Preserve reviewed paper-list order. Papers with no reviewed equations or failed HTML become `{}`. Write `data/final_data.json` atomically only after all validation succeeds.

## Orchestration and Artifacts

`run_pipeline.py` invokes all stages sequentially with no arguments. HTML remains a validated download cache; every derived stage is always rebuilt into a temporary directory and swapped into place only after validation. A failure stops downstream execution while preserving the last complete outputs.

Intermediate data is written under a new isolated root such as:

```text
data/pipeline/documents/
data/pipeline/chunks/
data/pipeline/symbols/
data/pipeline/evidence/
data/pipeline/meanings_raw/
data/pipeline/symbol_meanings/
data/pipeline/relations/
data/pipeline/meanings/
data/pipeline/reports/
```

The runner writes a consolidated build report containing input fingerprints, successful and skipped papers, unresolved equations, stage counts, runtimes, model/device information, rejection totals, and final coverage.

## Test Plan

- Mock arXiv success, cache reuse, corrupt HTML, retryable errors, and permanently missing HTML.
- Verify equation alignment precedence, ambiguity rejection, section hierarchy, sentence offsets, and cross-references.
- Test MathML composite-symbol reconstruction and LaTeX fallback for Greek, subscripts, superscripts, primes, and decorators.
- Verify deterministic BM25 ranking, same-paper isolation, source provenance, and query deduplication.
- Test meaning hard filters, MathBERT reranking, CPU fallback, and extractive-only output.
- Test symbol definition patterns, coordination, `respectively`, single-letter alias protection, and rejection reasons.
- Verify complete directed relation matrices and fixed grade/description invariants.
- Run an end-to-end fixture with one valid paper, one empty reviewed paper, and one failed-HTML paper.
- Assert deterministic paper/equation ordering and exact compatibility with the four-field final schema.
- Acceptance requires atomic output, zero unresolved aligned equations for successful papers, no unsupported non-empty values, and complete relations.

## Assumptions

- `data/3_equations.json` and annotations are reviewed inputs, not regenerated by this pipeline.
- Only successfully downloaded HTML papers are processed; failed papers remain as empty objects in the final export.
- Cached HTML is not redownloaded when valid, despite derived artifacts always being rebuilt.
- Equation MathBERT is inference-only and non-generative.
- Symbol extraction and symbol meanings are deterministic and rule-based.
- Existing source2 directories remain untouched until the flat pipeline reproduces and validates `final_data.json`.
