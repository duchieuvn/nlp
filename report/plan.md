# Report Revision Plan: Rubric-Focused Scientific System Discussion

## Summary

Rewrite the report so it is no longer mainly a code/pipeline description. The main body should answer the rubric directly:

1. **Key ideas**: what solution ideas solve equation extraction, meaning extraction, symbol meaning, relations, and auditability.
2. **Generalization**: why the approach should work on arbitrary quantum physics arXiv papers, with evidence from the produced dataset.
3. **Quality**: critically evaluate what worked, what failed, and why some limits could not be improved further.

Keep the existing pipeline diagram and a shortened System Architecture section, but make them supporting material rather than the center of the report.

## Recommended Report Structure

1. **Overview**
   - Keep `Problem Analysis` short.
   - Keep `Approach Overview`, but make it more conceptual:
     - parse arXiv HTML into structured paper objects,
     - use embedding space for semantic matching,
     - use deterministic rules for auditability,
     - avoid generative text because the dataset requires source-backed evidence.

2. **Key Ideas of the Approach**
   - This should be the most important section.
   - Do not describe files or code here.
   - Organize by required data item:
     - **Enumerated equations**: numbered display equations are selected because physics papers use equation numbers for important equations.
     - **Equation meaning**: combine local context, cross-references, BM25, and MathBERT embeddings to find explanatory source sentences.
     - **Symbol meaning**: extract symbols from MathML/LaTeX, then retrieve nearby definition sentences using aliases and definition patterns.
     - **Relations**: represent equation pairs as graph edges using cross-reference cues, shared symbols, section proximity, and semantic similarity.
     - **Audit trail**: store evidence and method decisions so every final entry is traceable.

3. **System Architecture**
   - Keep this section, but shorten it compared with the current version.
   - Explain stages 2-5 only as supporting infrastructure:
     - Stage 2 creates structured documents.
     - Stage 3 creates retrieval chunks.
     - Stage 4 selects dataset equations.
     - Stage 5 builds MathBERT embeddings.
   - Keep one small bordered JSON snippet per stage if space allows.
   - Avoid long code-level explanations such as exact function behavior unless needed to clarify an idea.

4. **Generalization**
   - Argue that the method generalizes because it relies on common arXiv/LaTeXML structure:
     - numbered equations,
     - MathML/LaTeX equation representations,
     - section and paragraph structure,
     - equation references such as `Eq. (1)`.
   - Argue that embeddings improve generalization because the system does not require exact wording.
   - Present evidence from the actual output:
     - `data/final_data.json` contains **100 paper keys**.
     - **54 papers** contain extracted equations.
     - **46 papers** correctly remain with empty equation dictionaries.
     - The dataset contains **353 equations**, satisfying the 350-equation target while finishing the last paper.
     - Each non-empty equation has required fields and an audit trail.
   - Mention limits:
     - papers with unusual HTML or missing equation labels are harder,
     - symbol definitions are often implicit,
     - retrieval quality depends on whether the paper states definitions in natural language.

5. **Quality and Critical Discussion**
   - Present strengths with evidence:
     - **Equation coverage**: 353 extracted equation entries.
     - **Meaning coverage**: 353/353 equations have non-empty meanings.
     - **Audit coverage**: 353/353 equations include audit-trail fields.
     - **Relations**: 2048 relation entries were generated; grades include strong, potential, and none.
   - Discuss weaknesses honestly:
     - Symbol definitions are the weakest part.
     - The analysis report found **1897 extracted symbols**, but only **202 defined symbols**.
     - **1695 symbols** had empty definitions, about **89.35%**.
     - Main rejection reason: no supported definition pattern in source text.
   - Explain why this could not be fully improved:
     - many physics symbols are defined implicitly by equations rather than prose,
     - some notation is reused or overloaded,
     - the project requires extractive, source-backed answers, so the system cannot invent missing definitions,
     - adding generative explanations would improve readability but violate traceability.
   - Discuss equation meanings critically:
     - all meanings are non-empty,
     - but the postprocessing report flagged **126 meanings** for review,
     - common issues include incomplete context, symbol-definition sentences selected as meanings, math-heavy spans, or too-short phrases.
   - Conclude that the approach is strong for auditable extraction and dataset construction, but weaker for complete symbol definition recovery.

6. **Conclusion**
   - Summarize the final position:
     - the system successfully builds the required dataset,
     - embedding-based retrieval makes it flexible,
     - deterministic rules and audit trails make it traceable,
     - remaining limitations come mainly from source-paper ambiguity and the extractive-only requirement.

## Concrete Edits to Make in `report/report.tex`

- Add a new section after `Approach Overview`:
  - `\section{Key Ideas of the Approach}`
- Keep the pipeline diagram page.
- Shorten the current `System Architecture` section:
  - remove overly detailed implementation paragraphs,
  - keep responsible files/output paths only if needed,
  - preserve bordered snippets only as compact examples.
- Add:
  - `\section{Generalization}`
  - `\section{Quality Discussion}`
  - `\section{Conclusion}`
- Use the following evidence numbers from the current outputs:
  - 100 paper keys,
  - 54 papers with equations,
  - 46 empty paper dictionaries,
  - 353 equations,
  - 353 non-empty meanings,
  - 353 audit trails,
  - 2048 relation entries,
  - 1897 extracted symbols,
  - 202 defined symbols,
  - 1695 undefined symbols,
  - 89.35% empty symbol definitions,
  - 126 flagged equation meanings.

## Test Plan

- Compile with:
  - `pdflatex -interaction=nonstopmode -halt-on-error report.tex`
- Run twice if LaTeX asks for outline/bookmark refresh.
- Check that:
  - the PDF compiles without errors,
  - the diagram still has its own page,
  - bordered snippets do not overflow margins,
  - the report clearly prioritizes rubric sections over code details,
  - all numerical evidence matches `data/final_data.json` and the analysis reports.

## Assumptions

- Keep the existing title and overall LaTeX style.
- Keep the pipeline diagram because it helps orient the reader.
- Keep System Architecture, but make it secondary.
- Do not add new experiments or change the dataset; only rewrite the report using existing evidence.
