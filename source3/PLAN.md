# Source3 Pipeline — Stages 6–10

## Current state (stages 1–5 complete)

```
data/pipeline/
  documents/<paper_id>.json     # parsed HTML: sections, paragraphs, sentences, MathML
  chunks/<paper_id>.json        # sentence / paragraph / cross_reference / neighborhood
  equations/<paper_id>.json     # reviewed equations aligned to raw DOM, with before/after sentences
  embeddings/<paper_id>.npz     # float32[n, 768] MathBERT vectors, L2-normalised
  embeddings/<paper_id>.json    # row metadata: vector_id, vector_kind, equation_id, chunk_id, sentence_id
```

### Embedding space layout (stage 5)

Stage 5 writes two families of vectors into the same `.npz` / `.json` pair:

**Family A — equation-centric** (existing): one vector per (equation × view), where
`vector_kind` ∈ `{"equation", "summary", "before_sentence", "after_sentence"}`.
These encode the equation's LaTeX plus context and are used by stage 6 for equation meaning retrieval.

**Family B — sentence-chunk** (new): one vector per sentence or neighborhood chunk in the
full paper, where `vector_kind = "sentence_chunk"`.
Input text is the raw chunk `text` field with no equation context prepended.
These are used by stage 8 for symbol definition search across the whole paper.

Row metadata fields common to both families:

| Field | Description |
|---|---|
| `vector_id` | Unique string key |
| `vector_kind` | One of the five kinds above |
| `equation_id` | Set for family A; `null` for family B |
| `chunk_id` | Set for family B; `null` for family A sentence-level rows |
| `sentence_id` | Set when the vector corresponds to a single sentence |
| `row` | Integer index into the `.npz` embeddings array |

Family B is built in one batched pass over `chunks/<paper_id>.json` immediately after
family A, using the same already-loaded model instance. Only chunks with
`chunk_type` ∈ `{"sentence", "raw_equation_neighborhood"}` are embedded.

Final output schema (`data/final_data.json`):

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

All meanings and definitions are verbatim spans from the paper. No text is generated.

---

## Stage 6 — Extract Equation Meanings `stage07_extract_meanings.py`

**Input:** `data/pipeline/{chunks,equations,embeddings}/<paper_id>.*`  
**Output:** `data/pipeline/meanings/<paper_id>.json`

All candidates are verbatim sentence spans from the paper.

### Candidate generation (ordered by priority)

1. **Cross-reference chunks** — chunks of type `cross_reference` whose `visible_equation_labels` matches the equation's anchor (`anchor_id`). These are sentences that explicitly cite the equation by label.
2. **Cue-pattern sentences** — scan `before_sentences` and the surrounding `raw_equation_neighborhood` chunk for sentences containing:
   - named-equation constructions: `"equation (N) ..."`, `"Eq. (N) states"`, `"(N) gives"`
   - descriptive openers before the equation: `"is defined as"`, `"is given by"`, `"represents"`, `"describes"`, `"denotes"`
   - appositive clauses immediately preceding the equation DOM node.
3. **Proximity fallback** — the single sentence directly before the equation (`before_sentences[0]`) if no candidates pass filters above.

### Hard filters (reject candidate if any holds)

- Purely procedural: starts with `"We"`, `"To"`, `"Note that"`, `"It follows"`.
- Math-heavy: latex token density > 40 % of word count.
- Symbol-definition clause: matches `"where X is"`, `"X denotes"`, `"let X"` — those belong in Stage 8.
- Incomplete fragment: fewer than 5 words, or ends mid-clause (trailing comma, colon, open paren).

### Reranking

Score each surviving candidate with a weighted sum:

| Feature                                                            | Weight |
| ------------------------------------------------------------------ | ------ |
| Cosine similarity to equation's `summary` vector                   | 0.5    |
| BM25 score against equation LaTeX as query                         | 0.2    |
| Source type bonus: `cross_reference` > `cue_pattern` > `proximity` | 0.2    |
| Proximity to equation (inverse document-order distance)            | 0.1    |

Use the equation's pre-built `summary` embedding (row where `vector_kind == "summary"`) for cosine similarity — no new inference needed.

Emit the top-scoring candidate verbatim as `meaning`. Emit `""` if no candidate survives filters.

Output schema:

```json
{
  "equation_id": "1",
  "meaning": "A qubit can be represented as a 2-dimensional vector with complex coefficients.",
  "source_sentence_id": "sec:section:1:p1:s1",
  "score": 0.83,
  "match_source": "cue_pattern"
}
```

---

## Stage 7 — Extract Symbols `stage06_extract_symbols.py`

**Input:** `data/pipeline/documents/<paper_id>.json`  
**Output:** `data/pipeline/symbols/<paper_id>.json`

For each reviewed equation, traverse its aligned MathML tree:

- Reconstruct composite identifiers through `msub`, `msup`, `msubsup`, `mover`, `munder`, and decorator nodes instead of collecting flat `<mi>` leaves.
- Exclude operators (`+`, `=`, `∑`, …), bare numbers, structural LaTeX commands, and single-use index variables (single letter appearing only in subscripts).
- For each symbol record: canonical name, original LaTeX forms, Unicode/Greek aliases, base identifier, and modifier stack.
- Fall back to a LaTeX tokenizer (regex over `\cmd`, `^`, `_`) when no usable MathML is present.

Output schema per equation:

```json
{
  "equation_id": "1",
  "symbols": [
    {
      "canonical": "psi_1",
      "latex_forms": ["\\psi_1", "\\psi_{1}"],
      "unicode": "ψ₁",
      "base": "psi",
      "modifiers": ["subscript:1"]
    }
  ]
}
```

Validate: no duplicate canonicals per equation, no empty symbol lists for equations with identifiable variables.

---

## Stage 8 — Extract Symbol Meanings `stage08_extract_symbol_meanings.py`

**Input:** `data/pipeline/{chunks,symbols,embeddings}/<paper_id>.*`  
**Output:** `data/pipeline/symbol_meanings/<paper_id>.json`

### Retrieval (embedding pre-filter)

For each symbol, build a query string from its aliases and LaTeX forms:

```
"where {alias} is  {alias} denotes  {latex_form}"
```

Embed this query with the same MathBERT model (loaded once per paper, reused across
all symbols). Compute cosine similarity against all `sentence_chunk` rows in the
paper's embedding space (family B from stage 5). Take the top **K = 30** sentences
as candidates. This replaces the previous full-scan over all sentence chunks.

Loading the embedding space is done once per paper; each symbol query is a single
matrix–vector multiply — no additional model inference per symbol.

### Pattern matching (on top-K candidates only)

Apply ordered high-precision rules to the retrieved candidate sentences:

| Priority | Pattern                              | Example                                                       |
| -------- | ------------------------------------ | ------------------------------------------------------------- |
| 1        | `where <sym> is` / `where <sym> are` | "where ψ₁ is the first component"                             |
| 2        | `<sym> denotes` / `<sym> represents` | "ψ denotes the state vector"                                  |
| 3        | `let <sym> be` / `let <sym> denote`  | "let N be the number of qubits"                               |
| 4        | Reverse: `<noun phrase> <sym>`       | "the amplitude ψ₁"                                            |
| 5        | Coordinated `respectively`           | "ψ₁ and ψ₂ are the first and second components, respectively" |

Match against `canonical`, `latex_forms`, and `unicode` aliases from stage 7.

If no pattern fires in the top-K, the symbol is omitted (no definition recorded).

### Hard filters (reject candidate span if any holds)

- Fewer than 2 meaningful words after the symbol.
- Contains another equation's LaTeX or a `\begin` block.
- Self-referential (span contains the equation's own LaTeX).
- Over 20 words (too vague).
- Stop-word-only after stripping the symbol.

Record: `canonical`, `definition` (verbatim span), `sentence_id`, `pattern`,
`confidence` (1.0–0.5 by rule rank), `retrieval_rank` (position in the top-K list).  
Omit symbols with no match found.

Output schema per equation:

```json
{
  "equation_id": "1",
  "symbol_meanings": {
    "psi_1": {
      "definition": "the first complex coefficient",
      "sentence_id": "sec:section:1:p2:s1",
      "pattern": "where_is",
      "confidence": 1.0
    }
  }
}
```

---

## Stage 9 — Build Relations `stage09_build_relations.py`

**Input:** `data/pipeline/{equations,chunks,embeddings,symbols}/<paper_id>.*`  
**Output:** `data/pipeline/relations/<paper_id>.json`

Create every directed pair `(A → B)` for all reviewed equations within a paper.

### Scoring features

| Feature                 | How                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **Explicit citation**   | A `cross_reference` chunk anchored at A mentions B's `anchor_id` or `visible_label`  |
| **Derivation cue**      | chunk text near A contains `"from (B)"`, `"using (B)"`, `"by (B)"`, `"follows from"` |
| **Equivalence cue**     | `"equivalent to"`, `"same as"`, `"reduces to"`, `"simplifies to"` between A and B    |
| **Special-case cue**    | `"special case of"`, `"when … reduces"`, `"setting … in"`                            |
| **Shared symbols**      | Jaccard overlap of canonical symbol sets ≥ 0.4                                       |
| **Section proximity**   | Same `section_id` = 1.0; adjacent section = 0.5; otherwise 0                         |
| **Semantic similarity** | Cosine between A's `summary` vector and B's `summary` vector                         |

### Grade mapping

Using fixed thresholds on the weighted sum:

| Grade       | Condition                                                                 |
| ----------- | ------------------------------------------------------------------------- |
| `strong`    | Any explicit citation / derivation / equivalence / special-case cue fires |
| `potential` | Shared symbols ≥ 0.4 OR semantic similarity ≥ 0.7 OR same section         |
| `none`      | Neither of the above                                                      |

### Fixed descriptions

Assign the **first** matching description in this order:

1. `"explicit citation"` — citation cue fired
2. `"derived from"` — derivation cue fired
3. `"equivalent"` — equivalence cue fired
4. `"special case"` — special-case cue fired
5. `"shares symbols"` — shared symbols ≥ 0.4
6. `"same section context"` — same section, no stronger signal

Output schema per paper:

```json
{
  "paper_id": "2401.11088",
  "relations": {
    "1": {
      "2": { "grade": "strong", "description": "explicit citation" },
      "3": { "grade": "potential", "description": "shares symbols" }
    }
  }
}
```

Every pair must have an entry. Self-pairs (`A → A`) are excluded.

---

## Stage 10 — Export Final Data `stage10_export_final.py`

**Input:** `data/pipeline/{equations,meanings,symbol_meanings,relations}/<paper_id>.json`  
**Output:** `data/final_data.json` (atomic)

For each paper in paper-list order:

1. Load equations, meanings, symbol meanings, relations.
2. Join on `equation_id`.
3. Build the final object:

```json
{
  "equation": "<latex>",
  "meaning": "<verbatim span or empty string>",
  "symbols": { "<canonical>": "<definition>" },
  "relations": { "<target_id>": { "grade": "...", "description": "..." } }
}
```

4. Validate:
   - `equation` text matches `3_equations.json`.
   - All reviewed equations present even if `meaning` is `""`.
   - `symbols` keys are unique canonicals from Stage 6.
   - Every other reviewed equation in the paper appears in `relations`.
   - Grade is one of `strong`, `potential`, `none`.
   - Description is one of the six fixed strings.

5. Papers with failed HTML or no reviewed equations → `{}`.

6. Write atomically via temp file + replace.

---

## File layout after all stages

```
data/pipeline/
  symbols/                   # stage06
  meanings/                  # stage07
  symbol_meanings/           # stage08
  relations/                 # stage09
data/final_data.json         # stage10
```

## Orchestration

Add stages 6–10 to `main.py` sequentially after stage 5. Each stage rebuilds into a temp directory and swaps atomically on success.
