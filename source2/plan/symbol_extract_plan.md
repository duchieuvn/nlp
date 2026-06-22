# Neural Symbol Meaning Extraction Plan

## Summary

Create `source2/plan/symbol_meaning_encoder_plan.md` describing a backward-compatible hybrid pipeline:

1. Preserve high-confidence regex definitions.
2. Use MathBERT bi-encoder reranking for unresolved symbols.
3. Use a weakly supervised MathBERT cross-encoder to classify symbol-to-phrase relations.
4. Preserve original LaTeX in all model inputs; use canonical names only for joins.
5. Leave `definition` empty whenever confidence or relation assignment is ambiguous.

## Implementation Design

- Build an inline-math index from original HTML. Canonicalize each `<math>` element while retaining its original LaTeX, sentence, paragraph, and DOM location.
- Parse every target into base, subscript, superscript, decorators, original `latex_forms`, aliases, and containing equation.
- Retrieve broadly with BM25, then rerank sentences using the existing MathBERT encoder with a representation containing original LaTeX, parsed components, equation, section, and nearby context.
- Install `en_core_web_sm` and generate extractive candidates from noun chunks, appositions, copular complements, and coordinated phrases.
- Fine-tune `witiko/mathberta` as a joint cross-encoder over symbol context and candidate phrase/context pairs.
- Use relation labels:
  - `DEFINES_COMPLETE_SYMBOL`
  - `DEFINES_BASE`
  - `QUALIFIES_SUBSCRIPT`
  - `QUALIFIES_SUPERSCRIPT`
  - `RELATED_NOT_DEFINITION`
  - `NO_RELATION`
- Bootstrap weak labels from existing regex definitions, HTML inline-symbol matches, modifier cues, coordinated left/right constructions, and hard negatives from wrong symbols in the same sentence or equation.
- Split training, validation, and test sets by paper to prevent context leakage.

## Selection And Output

- Existing regex definitions take precedence.
- Neural extraction runs only for unresolved symbols.
- Populate the existing flat `definition` only from a source span classified as `DEFINES_COMPLETE_SYMBOL`, or `DEFINES_BASE` when the target has no semantic modifiers.
- Store component relations under `audit`; do not synthesize a definition by joining unrelated source phrases.
- Record model versions, original LaTeX, parsed components, candidate phrase, full evidence sentence, relation probabilities, bi-encoder score, cross-encoder score, source IDs, and rejection reason.
- Select thresholds on the reviewed benchmark for at least 90% precision. Reject predictions below threshold or with less than a 0.10 margin over the competing relation.
- During evaluation, write predictions to a separate comparison directory. Promote them into `data/source2/symbol_meanings` only after the acceptance gates pass.

## Evaluation

- Create a stratified reviewed benchmark covering current regex successes, `no_retrieved_alias`, `no_supported_definition_pattern`, decorated symbols, subscripts, superscripts, and coordinated phrases.
- Require at least 90% precision for accepted complete definitions and measurable recovery over the current 202 definitions.
- Report per-relation precision, recall, F1, accepted coverage, remaining empty rate, and false assignments between symbols in the same sentence.
- Include focused cases such as `u^{L}_{n}` versus `u^{R}_{n}`, ensuring “left chain” and “right chain” map to the correct modifiers without inventing a complete definition.

## Tests And Assumptions

- Test original-LaTeX preservation, canonical HTML matching, component parsing, paper-level data splitting, noun-phrase extraction, hard-negative generation, threshold abstention, and audit serialization.
- Keep the current public `SymbolMeaningRecord` fields unchanged; component relations remain audit metadata.
- MathBERT remains extractive and non-generative.
- Ambiguous component-only evidence is useful for auditing but does not populate `definition`.
