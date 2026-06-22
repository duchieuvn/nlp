# Pipeline 2 Build Plan

Pipeline 2 is a retrieval-augmented extraction pipeline for building an
equation knowledge graph. Retrieval selects evidence from each paper, while
deterministic rules produce the final values. No language model or other text
generator writes meanings, symbol definitions, or relation descriptions.

## 1. Prepare Inputs

- Read the target paper IDs, equation records, annotations, and cached arXiv
  HTML files.
- Check that every paper containing target equations has an HTML source.
- Record missing or empty papers in the build report instead of silently
  dropping them.

**Input:** equation JSON, annotation JSON, and arXiv HTML files.  
**Output:** validated paper-level build jobs.

## 2. Build Structured Paper Documents

- Parse each HTML document with BeautifulSoup.
- Preserve the document hierarchy: title, sections, paragraphs, sentences,
  equations, and explicit equation references.
- Resolve target equations using annotation IDs first, then visible equation
  labels and normalized LaTeX fallbacks.
- Attach the nearest previous and next paragraphs to every equation.

**Output:** one structured document per paper in
`data/source2/structured_papers/<paper_id>.json`.

## 3. Validate the Document Structure

- Verify that paper, section, paragraph, sentence, equation, and reference IDs
  are unique.
- Check parent-section links, sentence offsets, equation-to-section links, and
  cross-reference targets.
- Stop the build when the structured representation is incomplete or
  inconsistent.

**Output:** trusted structured documents for downstream retrieval.

## 4. Create Multiple Chunk Views

Add a `chunks.py` module and create several views over each structured paper:

- **Sentence chunks:** precise evidence for meanings and symbol definitions.
- **Paragraph chunks:** the main lexical retrieval unit.
- **Equation-neighborhood chunks:** section title, previous paragraph,
  equation LaTeX, and next paragraph.
- **Section-aware chunks:** add section titles to improve scientific context.
- **Cross-reference chunks:** sentences that explicitly cite equation numbers.

Every chunk should retain `chunk_id`, `paper_id`, `chunk_type`, text, section
metadata, paragraph and sentence IDs, nearby equation IDs, and symbols.

**Output:** auditable chunks that always point back to source text.

## 5. Extract and Normalize Equation Symbols

- Parse candidate symbols from each equation's LaTeX.
- Normalize Greek, Unicode, decorated, and subscripted forms for matching.
- Exclude structural LaTeX commands, operators, and likely index variables.
- Store symbol aliases for retrieval without assigning definitions yet.

**Output:** normalized symbol candidates and search forms per equation.

## 6. Build the Retrieval Layer

Add a `retrieval.py` module with BM25 as the primary retriever.

- Index sentence, paragraph, and equation-neighborhood chunks.
- Preserve math tokens and chunk metadata during tokenization.
- Support paper, section, chunk-type, equation, and symbol filters.
- Return ranked chunk IDs, scores, types, and source text.
- Add TF-IDF as a reproducible comparison baseline.

Optional later experiment: rerank BM25 candidates with
`all-MiniLM-L6-v2`. Embeddings may only score candidates; they must not
generate final text.

**Output:** ranked evidence candidates for each extraction task.

## 7. Retrieve Meaning Evidence

- Build deterministic queries from the equation number, important symbols,
  and cues such as `defines`, `describes`, `represents`, and `gives`.
- Search equation-neighborhood, paragraph, and sentence chunks from the same
  paper.
- Keep the top candidates and their BM25 scores for extraction and auditing.

**Output:** ranked source sentences that may explain each equation.

## 8. Extract Equation Meanings

- Match fixed patterns such as `Equation (N) describes ...` and named-equation
  phrases.
- Rerank the BM25 top-k candidate chunks or sentences with MathBERT. Encode a
  deterministic equation representation (LaTeX, important symbols, section
  title, and immediate context) and each candidate, then use cosine similarity
  only as an additional ranking feature.
- Combine normalized MathBERT similarity with the existing explicit equation
  citation, definition cue, proximity, section match, and BM25 features. Do not
  let MathBERT similarity override hard rejection rules or act as confidence by
  itself.
- Prefer a complete source sentence; only shorten it with a documented
  mechanical rule.
- After extraction, run the conservative `postprocessing` stage. It may remove
  a trailing LaTeX-heavy appositive followed by a rephrasing marker such as
  `in other words:`, but only when the retained source span is a complete
  descriptive clause. It may also extract the literal claim from a causal
  reporting construction such as `Since ..., one observes that ...`. Preserve
  the original source text and audit the removal. Write processed copies under
  `data/postprocessing/equation_meanings/`; never overwrite extraction output.
- Reduce meanings to extractive noun phrases of at most 12 natural-language
  words using ordered subject, object, named-complement, and science-head rules.
  Exclude symbol-definition and procedural clauses; leave the meaning empty
  when no reliable phrase remains.
- Keep MathBERT non-generative: it selects among source chunks but never writes
  or paraphrases the equation meaning.
- Return an empty meaning when no candidate reaches the confidence threshold.
- Log the model version, MathBERT score, component feature scores, candidate
  rank before and after reranking, and selected source IDs.

**Output:** an extractive meaning plus source evidence and method metadata.

## 9. Retrieve and Extract Symbol Definitions

- Query each normalized symbol with fixed definition cues such as `denotes`,
  `represents`, `is defined as`, and `where`.
- Apply high-precision regex and phrase patterns to retrieved sentences.
- Use spaCy dependency patterns only as a controlled fallback for copular and
  appositive definitions.
- Reject vague, overly long, or math-heavy spans and omit low-confidence
  definitions.

**Output:** symbol-to-definition mappings copied from paper text.

## 10. Classify Equation Relations

- Create every required equation pair within a paper.
- Gather explicit cross-references, derivation cues, equivalence or
  special-case cues, shared symbols, section distance, and context similarity.
- Convert the evidence into `strong`, `potential`, or `none` with fixed scoring
  thresholds.
- Select descriptions only from a fixed vocabulary such as
  `explicit citation`, `derived from`, `equivalent`, `special case`,
  `shares symbols`, and `same section context`.

**Output:** a complete deterministic relation map for every equation.

## 11. Record the Audit Trail

For every retained value, record:

- source paper, chunk, paragraph, and sentence IDs;
- matched source sentence or span;
- retrieval score and distance to the equation;
- extraction pattern or classification rule;
- confidence score and selected final value.

The audit trail must demonstrate that retrieval selected evidence and rules
produced the answer.

## 12. Validate and Export Final Results

- Validate the required fields: `equation`, `meaning`, `symbols`, `relations`,
  and `audit-trail`.
- Ensure every relation includes a valid grade and description and covers all
  other relevant equations in the paper.
- Reject unsupported non-empty values or values without audit evidence.
- Write one atomic JSON file per paper and print aggregate build statistics.
- Build the strict combined export at `data/final_data.json`. Each equation has
  exactly `equation`, `meaning`, `symbols`, and `relations`; `symbols` is a
  canonical-name-to-definition object, and audit metadata is excluded for now.
  Preserve the normalized paper order from `data/paper_list_46.txt`.

**Output:** final equation knowledge-graph JSON files and a build report.

## 13. Evaluate and Tune

- Build a small hand-labeled gold set for meanings, symbols, relations, and
  evidence chunks.
- Compare sentence, paragraph, equation-neighborhood, section-aware, and
  sliding-window chunking.
- Compare TF-IDF, BM25, field-weighted BM25, and optional BM25 plus embedding
  reranking.
- Report top-1 evidence accuracy, top-5 evidence recall, extraction precision,
  relation precision, runtime, and common error categories.
- Select thresholds from validation results, not from the test set.

## Recommended Implementation Order

1. Finish and test structured-document generation and validation.
2. Implement chunk models and chunk builders.
3. Implement BM25 retrieval and deterministic query templates.
4. Implement meaning extraction.
5. Implement symbol extraction.
6. Implement pairwise relation classification.
7. Add audit records and final-schema validation.
8. Run ablations, tune thresholds, and document limitations.

The recommended production baseline is DOM-aware chunking, BM25 retrieval,
regex and spaCy rule-based extraction, fixed-label relation classification,
and a complete audit trail. TF-IDF and embedding reranking should remain
comparison experiments until they show a measurable improvement.
