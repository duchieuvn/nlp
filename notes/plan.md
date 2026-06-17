# RULES

Latex is always not allowed

# Pipeline Plan

Use a deterministic, evidence-based extraction pipeline. The core idea is to rely on the paper structure itself rather than any generated text: first identify numbered equations, then inspect the surrounding source text to infer meaning, symbols, and relations. This is the safest approach under the no-prompt constraint because it can be justified directly from the arXiv source.

1. Read the assigned paper list and preserve the exact order.
   - Normalize each line to a clean arXiv ID.
   - Keep the original order because the dataset must follow the provided list exactly.

2. Acquire paper sources from arXiv only.
   - Prefer HTML when available because the equation structure and surrounding text are easier to parse.
   - Fall back to PDF source when HTML is unavailable or incomplete.
   - Respect robots.txt and use conservative delays between requests.
   - Cache downloaded sources so the same paper is not fetched repeatedly.

3. Convert each paper into a structured document representation.
   - Split the paper into sections, paragraphs, displayed equations, captions, and reference fragments.
   - Preserve the order of the source text so each equation can be inspected in local context.
   - Keep numbering markers such as `(1)` or `Eq. (1)` attached to the extracted equation.

4. Detect enumerated equations.
   - Identify display equations that carry an equation number in brackets.
   - Ignore unnumbered display math unless it is clearly tied to a numbered equation.
   - Store the LaTeX form together with the visible equation number exactly as it appears in the paper.

5. Select the relevant equations for the dataset.
   - Keep the first 7 relevant enumerated equations per paper.
   - If a paper contains fewer than 7 enumerated equations, keep all of them.
   - Continue paper by paper until the dataset reaches the required total range of 350 to 356 equations.
   - Do not stop mid-paper; once a paper is selected, process it completely.

6. Extract the meaning of each equation.
   - Read the sentence immediately before and after the equation first.
   - Look for cue phrases such as “where”, “is given by”, “defines”, “describes”, “the Hamiltonian”, or “the wave function”.
   - Use section titles and nearby definitions as secondary evidence when the local sentence is not enough.
   - Prefer short labels over long explanations so the meaning stays compact and reusable.

7. Extract symbols and abbreviations.
   - Scan the local context for explicit definitions of variables, abbreviations, operators, and named quantities.
   - Record only paper-specific symbols, not standard mathematical operators.
   - Link a symbol to its defining phrase when the paper states one directly, or to the nearest clear explanatory text when the definition is implicit but strong.
   - If a symbol cannot be justified from source text, leave it out rather than guessing.

8. Build equation relations within the same paper.
   - Compare each equation with every other selected equation from the paper.
   - Use shared symbols, repeated phrases, equation references, and surrounding verbs such as “from”, “follows”, “equivalent”, “special case”, “generalization”, or “limit”.
   - Treat direct textual references and explicit derivations as strong relations.
   - Treat weaker thematic or contextual overlap as potential relations.
   - Mark unrelated pairs as none.

9. Apply a simple relation grading rule.
   - `strong`: the paper explicitly states or clearly implies the relation.
   - `potential`: the equations are likely connected, but the source text does not fully prove it.
   - `none`: there is no meaningful relation supported by the local evidence.
   - This rule set should be easy to explain in the documentation and should work consistently across papers.

10. Produce an audit trail for every equation.
    - Log which extraction method found the equation.
    - Log which context sentence supported the meaning.
    - Log which text fragments supported each symbol definition.
    - Log which evidence supported each relation classification.
    - Keep the audit trail short, method-based, and readable.

11. Assemble the JSON output.
    - Use the arXiv ID as the top-level key.
    - Use equation numbers as the nested keys.
    - Store `equation`, `meaning`, `symbols`, `relations`, and `audit-trail` for each equation.
    - Ensure papers with no enumerated equations still appear with an empty equation dictionary.

12. Validate the result before saving.
    - Check ordering, total equation count, and the seven-equation-per-paper limit.
    - Check that the JSON structure matches the required schema.
    - Spot-check examples to confirm that meanings, symbols, and relations are evidence-based.

13. Document the method.
    - Explain why the structural, rule-based approach was chosen.
    - Explain how the approach generalizes to different quantum physics papers.
    - Discuss weaknesses honestly, especially cases where equations are not clearly defined or where symbols are introduced far from the equation.
    - Show how the audit trail supports quality analysis and debugging.

Suggested implementation stages:

- Stage 1: source acquisition and caching
- Stage 2: paper parsing and equation detection
- Stage 3: meaning extraction from local context
- Stage 4: symbol extraction from definition cues
- Stage 5: relation graph construction and grading
- Stage 6: audit trail assembly and JSON export
- Stage 7: validation and documentation
