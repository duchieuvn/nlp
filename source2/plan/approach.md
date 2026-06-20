# Approach v2: Extractive Retrieval Pipeline for the Equation Knowledge Graph

This document describes a safer and stronger approach for the NLP project:
use advanced chunking plus classical retrieval, especially BM25, to find
candidate evidence, then extract the final JSON values with deterministic
rules. The core principle is:

> Retrieval may select evidence, but it must not write the answer.

This keeps the solution compatible with the assignment rule against prompting
and text generation. Every final value must be copied from, normalized from, or
classified based on arXiv paper text and must be traceable in the audit trail.

V2 is a retrieval-augmented extraction pipeline:

1. Structure each paper into sections, paragraphs, sentences, equations, and cross-references.
2. Create multiple evidence chunks around equations and symbols.
3. Use BM25 to retrieve relevant source passages.
4. Optionally rerank passages with sentence embeddings.
5. Extract meanings, symbol definitions, and relations using deterministic rules.
6. Assign fixed relation labels and preserve evidence in the audit trail.

## Compliance Boundary

Allowed:

- BM25, TF-IDF, and other lexical retrieval methods.
- Sentence embeddings such as `all-MiniLM-L6-v2`, used only for similarity,
  clustering, or reranking.
- Cross-encoders or neural rerankers only if they output scores/classes and do
  not generate text. This is more difficult to defend, so treat it as an
  experiment rather than the main method.
- Regex, gazetteers, rule-based matching, dependency parsing, and finite label
  classification.
- Deterministic relation descriptions from a fixed vocabulary such as
  `"explicit citation"`, `"derived from"`, `"shares symbols"`, `"same context"`.

Not allowed:

- Prompting ChatGPT, Claude, Perplexity, local LLMs, or any text generator to
  write meanings, symbol definitions, or relation descriptions.
- Generating summaries from retrieved chunks.
- Filling missing values with model guesses.
- Using external APIs for extraction.

Recommended final stance in the report:

> BM25 and optional embeddings are used only to locate relevant paper text.
> The final values are extracted from this text with deterministic rules and
> every selected value is logged in the audit trail.

## Recommended Tech Stack

| Layer                    | Primary choice                                                     | Why                                                                    |
| ------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| HTTP acquisition         | `requests`                                                         | Simple, deterministic fetching with custom User-Agent and crawl delay. |
| HTML parsing             | `beautifulsoup4` + `lxml`                                          | Robust DOM navigation for arXiv LaTeXML HTML.                          |
| PDF fallback             | `PyMuPDF`                                                          | Practical text extraction fallback when HTML is missing.               |
| Equation extraction      | BeautifulSoup DOM + MathML annotation/alttext + fallback converter | Best chance of recovering LaTeX without using arXiv source downloads.  |
| Sentence splitting       | spaCy sentencizer or rule-based splitter                           | Needed for sentence-level evidence selection.                          |
| Chunking                 | DOM-aware, section-aware, equation-neighborhood chunks             | Gives retrieval structured evidence instead of arbitrary text windows. |
| Lexical retrieval        | BM25 via `bm25s` or `rank-bm25`                                    | Strong, transparent, non-generative baseline.                          |
| Baseline retrieval       | `scikit-learn` `TfidfVectorizer`                                   | Useful for ablation and comparison.                                    |
| Optional dense retrieval | `sentence-transformers/all-MiniLM-L6-v2`                           | Encoder-only semantic similarity; no text generation.                  |
| Extraction               | regex + spaCy Matcher/PhraseMatcher + dependency parsing           | Deterministic span extraction from candidate text.                     |
| Validation               | `jsonschema` or `pydantic`                                         | Enforces the final required schema.                                    |
| Evaluation               | small gold set + precision/recall/F1 + error analysis              | Gives evidence for report claims.                                      |

## Proposed Architecture

```text
paper_list_44.txt
  -> fetch arXiv HTML/PDF with cache and crawl delay
  -> extract numbered equations and surrounding DOM context
  -> build structured document model
  -> create several chunk views
  -> index chunks with BM25 and optional embedding vectors
  -> retrieve candidate evidence per equation / symbol / relation pair
  -> extract final values with deterministic rules
  -> score and select best candidate
  -> write final JSON and audit trail
```

## Structured Document Model

Before retrieval, convert each paper into a structured representation:

```json
{
  "arxiv_id": "2401.13506",
  "sections": [
    {
      "section_id": "S2",
      "title": "Model",
      "paragraphs": [
        {
          "paragraph_id": "S2.P3",
          "text": "...",
          "equation_refs": ["1", "2"],
          "symbols_mentioned": ["H", "psi"]
        }
      ],
      "equations": [
        {
          "eq_num": "1",
          "latex": "...",
          "context_before": "...",
          "context_after": "..."
        }
      ]
    }
  ]
}
```

This helps chunking, retrieval, relation classification, and auditability.

## Advanced Chunking Strategies

Use multiple chunk types instead of one fixed window. Each chunk should carry
metadata: `chunk_id`, `arxiv_id`, `section_title`, `section_id`,
`paragraph_id`, `eq_nums_nearby`, `symbols`, and `source`.

### 1. Equation-Neighborhood Chunks

Best default chunk for meaning and relations.

Content:

```text
section heading
previous paragraph
equation number and LaTeX
next paragraph
```

Why it helps:

- Equation meanings are usually introduced before or after the equation.
- Relation cues often occur in nearby prose.
- Easy to audit because the chunk is local and human-readable.

Experiment parameters:

- Previous/next 1 paragraph.
- Previous/next 2 paragraphs.
- Previous 2 + next 1 paragraphs.
- Sentence window of +/- 3, +/- 5, +/- 8 sentences.

### 2. Sentence Chunks

Best for final value extraction.

Content:

```text
single sentence
```

Why it helps:

- Meaning values can be selected as whole source sentences.
- Symbol definitions often fit in one sentence.
- BM25 scores are less diluted than on long paragraphs.

Risk:

- Some definitions span two sentences, so keep neighbor sentence IDs.

### 3. Paragraph Chunks

Best BM25 retrieval baseline.

Content:

```text
one HTML paragraph
```

Why it helps:

- Paragraphs are natural author-written units.
- Less noisy than full sections.
- Works well with BM25 length normalization.

### 4. Section-Aware Chunks

Best for relation and fallback meaning.

Content:

```text
section title + paragraph or section title + equation neighborhood
```

Why it helps:

- Section titles such as "Hamiltonian", "Model", "Master equation", or
  "Dynamics" are strong context signals.
- Helps BM25 when the equation context itself is short.

### 5. Symbol-Centered Chunks

Best for symbol definitions.

For each symbol in an equation, create a query-specific search over chunks
containing one of the symbol text forms:

- `H`, `$H$`, `\(H\)`
- `psi`, `\psi`, Unicode equivalents if present
- subscripted forms like `omega_c`, `omega_0`, `E_n`

This is not a permanent chunk type; it is a filtered retrieval view.

### 6. Cross-Reference Chunks

Best for relation extraction.

For each equation number, collect sentences that mention:

- `Eq. (N)`
- `Equation (N)`
- `Eqs. (N)-(M)`
- bare `(N)` when it is likely an equation reference

These chunks provide the highest-confidence evidence for `strong` relation
edges.

### 7. Sliding Window Chunks

Useful as an experiment, not the main method.

Settings to test:

- 100 words with 30-word overlap.
- 150 words with 50-word overlap.
- 250 words with 75-word overlap.

Weakness:

- Less interpretable than DOM-aware chunks.
- Can cut definitions in awkward places.

## Retrieval Methods to Try

### Method A: BM25 Baseline

Use BM25 over paragraph chunks and equation-neighborhood chunks.

Candidate libraries:

- `bm25s`: fast modern Python BM25 implementation using NumPy/SciPy.
- `rank-bm25`: simple and easy to inspect.
- Pyserini/Lucene: strong IR toolkit, but heavier because it needs Java.

Suggested BM25 settings:

```text
k1 = 1.2 or 1.5
b = 0.75
top_k = 5 or 10
lowercase = true
keep math tokens = true
```

Why BM25 is strong for this project:

- It is non-generative.
- It is explainable: matching terms and scores can be logged.
- It works well when queries contain exact terms like equation numbers,
  symbols, and definition cue words.

### Method B: BM25F / Field-Weighted BM25

Treat chunks as structured fields:

```text
title weight: 2.0
equation LaTeX tokens weight: 1.5
nearby prose weight: 1.0
section title weight: 1.5
```

This is useful because scientific papers have structure. A match in a section
title or equation-neighborhood text may be stronger than a match in a random
paragraph.

If not using a BM25F library, implement a simple weighted score:

```text
score = 2.0 * bm25(title)
      + 1.5 * bm25(equation_text)
      + 1.0 * bm25(prose)
      + 1.5 * bm25(section_title)
```

### Method C: TF-IDF Baseline

Use `TfidfVectorizer` as a reproducibility baseline.

Suggested settings:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.85,
    norm="l2",
)
```

Why keep it:

- It is easy to explain.
- It gives an ablation baseline against BM25.
- It can rank candidate meaning sentences by corpus-specific terms.

### Method D: Hybrid BM25 + Dense Embeddings

Use BM25 for first-stage retrieval and `all-MiniLM-L6-v2` only for reranking
or tie-breaking.

Safe usage:

```text
BM25 top 20 chunks
  -> encode candidate chunks with all-MiniLM-L6-v2
  -> encode deterministic query/context
  -> cosine similarity rerank
  -> final extraction still done by regex/rules
```

Important limits:

- Do not ask the model to answer.
- Do not use prompts that request generation.
- Do not synthesize a new meaning sentence.
- Log model name, query text, chunk ID, and similarity score.

Why this can be useful:

- BM25 misses paraphrases.
- Embeddings can retrieve semantically similar context when exact words differ.
- `all-MiniLM-L6-v2` is intended for sentence and short paragraph encoding;
  avoid overly long chunks because long input is truncated.

## Query Design

Use deterministic query templates.

### Meaning Queries

For equation `N`:

```text
"Equation (N)" "Eq. (N)" meaning represents describes gives defines called known as
```

Also include nearby symbols:

```text
H psi omega Hamiltonian wave function energy
```

Recommended query string:

```text
Eq N Equation N defines describes represents gives called known as <top symbols>
```

### Symbol Queries

For symbol `S`:

```text
S where denotes represents is defined as called corresponds to
```

For Greek symbols, search multiple forms:

```text
psi \psi phi \phi alpha \alpha omega \omega
```

### Relation Queries

For equation pair `(N, M)`:

```text
Eq N Eq M using substituting inserting follows from derived from gives yields equivalent special case
```

For equation `N` alone:

```text
Eq N using substituting follows from derived from Eq Equation
```

## Final Value Extraction Methods

Retrieval gives candidate chunks. The final value comes from the best
candidate sentence/span selected by deterministic extraction.

### Meaning Extraction

Output should be either:

- a full sentence from the paper, or
- a mechanically shortened phrase from a matched pattern.

Safer default:

> Use the full source sentence as `meaning`.

Candidate patterns:

```text
Equation (N) describes <X>
Equation (N) gives <X>
Eq. (N) defines <X>
Eq. (N) represents <X>
<X> is called the <Y>
<X> is known as the <Y>
the <Y> equation
the <Y> Hamiltonian
the <Y> wave function
```

Meaning candidate scoring:

| Feature                                                | Weight |
| ------------------------------------------------------ | -----: |
| Sentence explicitly mentions equation number           |     +5 |
| Contains `defines`, `describes`, `represents`, `gives` |     +3 |
| Contains named-equation cue                            |     +3 |
| In previous/next paragraph of equation                 |     +2 |
| Same section as equation                               |     +1 |
| High BM25 score                                        |     +1 |
| Sentence too short or too long                         |     -1 |

Fallback:

- If no sentence is reliable, return empty string.
- Do not invent a description.

Audit example:

```json
"meaning": "eq=3 strategy=citation_sentence bm25=8.42 chunk=S2.P4 sentence='Equation (3) gives the effective Hamiltonian ...'"
```

### Symbol Extraction

Step 1: extract possible symbols from LaTeX.

Keep:

- Greek commands: `alpha`, `beta`, `gamma`, `psi`, `omega`, etc.
- important uppercase letters: `H`, `E`, `L`, `N`, `S`, `U`, etc.
- meaningful lowercase variables if not obvious indices.
- decorated/subscripted forms where needed: `omega_c`, `E_n`, `H_0`.

Skip:

- structural LaTeX commands: `frac`, `left`, `right`, `begin`, `end`,
  `mathrm`, `mathbf`, `mathcal`, `sqrt`, `sum`, `int`.
- operators: `cdot`, `times`, `partial`, `nabla`, `leq`, `rightarrow`.
- likely indices unless explicitly defined: `i`, `j`, `k`, `l`, `m`, `n`.

Step 2: search retrieved chunks for definition patterns.

High-value regex patterns:

```text
where S is <definition>
where S denotes <definition>
where S represents <definition>
S is the <definition>
S denotes the <definition>
S represents the <definition>
S is defined as <definition>
let S be <definition>
with S being <definition>
S, the <definition>
the <definition> S
```

Definition span filters:

```text
1 to 15 words
stop at comma, semicolon, period, "and", or equation reference
reject spans containing too many math symbols
reject spans that are only "state", "term", "quantity" unless modified
```

Use spaCy dependency parsing as fallback for:

- copular definitions: `H is the Hamiltonian`
- appositions: `H, the system Hamiltonian`
- noun phrases: `the coupling strength g`

Audit example:

```json
"symbol": "sym='H' pattern='where_denotes' chunk=S2.P3 text='where H denotes the system Hamiltonian' value='system Hamiltonian'"
```

### Relation Extraction

The assignment requires every equation to include a relation to every other
relevant equation in the same paper:

```json
"relations": {
  "2": {"grade": "strong", "description": "derived from Eq. (2)"},
  "3": {"grade": "potential", "description": "shares symbols H, psi"},
  "4": {"grade": "none", "description": ""}
}
```

Use a two-level method:

1. Extract evidence.
2. Convert evidence to `grade` and `description` using fixed rules.

Strong evidence:

- explicit citation: `Eq. (M)` appears near equation `N`.
- derivation cue with citation: `using`, `substituting`, `inserting`,
  `from`, `follows from`, `therefore`, `hence`, `which gives`.
- equivalence cue: `equivalent to`, `same as`, `reduces to`.
- special-case cue: `in the limit`, `for zero temperature`, `when`.

Potential evidence:

- shared symbol overlap above threshold.
- same section and close equation distance.
- high BM25/TF-IDF/embedding similarity between equation-neighborhood chunks.
- same named physical concept appears in both contexts.

None:

- no explicit citation.
- weak or no shared symbols.
- distant sections and low context similarity.

Suggested scoring:

```text
score = 5 * explicit_cross_reference
      + 4 * derivation_cue_between_pair
      + 3 * equivalence_or_special_case_cue
      + 2 * shared_symbol_jaccard
      + 1 * same_section
      + 1 * context_similarity
      - 1 * section_distance
```

Decision:

```text
score >= 5: strong
score >= 2: potential
otherwise: none
```

Relation description should come from a fixed label map:

```text
explicit_cross_reference -> "explicit citation"
substituting/using/from -> "derived from"
equivalent/same as -> "equivalent"
reduces to/in limit -> "special case"
shared symbols only -> "shares symbols <list>"
same section only -> "same section context"
```

This is classification, not generation, because the output descriptions come
from a fixed deterministic vocabulary.

Audit example:

```json
"relation": "eq=3 other=2 grade=strong evidence='substituting Eq. (2)' chunk=S2.P8 score=7 description='derived from'"
```

## Candidate Selection and Confidence

For every extracted value, store an internal confidence object:

```json
{
  "value": "system Hamiltonian",
  "source_sentence": "where H denotes the system Hamiltonian",
  "chunk_id": "2401.13506:S2.P3",
  "method": "regex_where_denotes",
  "bm25_score": 9.21,
  "distance_to_equation": 1,
  "confidence": 0.86
}
```

Only write the final required JSON fields, but include the useful details in
the audit trail.

Recommended confidence thresholds:

```text
meaning >= 0.50: keep
symbol >= 0.65: keep
strong relation >= 0.70: strong
potential relation >= 0.35: potential
otherwise: none
```

If confidence is too low, leave the meaning empty or omit the symbol. For
relations, still include every pair, but use grade `none`.

## Experiments to Run

### Experiment 1: Chunking Ablation

Compare retrieval quality for:

- paragraph chunks
- sentence chunks
- equation-neighborhood chunks
- section-aware chunks
- sliding windows

Metrics:

- manual top-1 evidence accuracy
- top-5 evidence recall
- final symbol precision
- final meaning usefulness

Expected result:

Equation-neighborhood chunks should work best for meaning and relations;
sentence chunks should work best for exact symbol definitions.

### Experiment 2: Retriever Ablation

Compare:

- TF-IDF
- BM25
- BM25F-style weighted fields
- BM25 + all-MiniLM reranking

Metrics:

- top-5 chunk contains usable evidence
- final extraction precision
- runtime
- audit clarity

Expected result:

BM25 should be the strongest default. Embeddings may improve paraphrase cases
but may be less transparent.

### Experiment 3: Symbol Extraction Modes

Compare:

- regex only
- regex + gazetteer
- regex + spaCy dependency parsing
- regex + dependency parsing + BM25 retrieval

Metrics:

- number of symbols extracted
- precision on a hand-labeled sample
- false positive categories

Expected result:

Regex gives highest precision. Dependency parsing improves recall but needs
filters to avoid noisy definitions.

### Experiment 4: Relation Evidence Ablation

Compare:

- cross-reference only
- cross-reference + derivation cues
- add shared symbols
- add context similarity

Metrics:

- strong relation precision
- potential relation usefulness
- percentage of `none` relations

Expected result:

Cross-reference and derivation cues should define `strong`. Shared symbols and
similarity are better for `potential`.

## Recommended Implementation Plan

### Phase 1: Improve Chunk Model

Add a `chunks.py` module:

```python
class Chunk:
    chunk_id: str
    arxiv_id: str
    chunk_type: str
    text: str
    section_title: str
    eq_nums_nearby: list[str]
    paragraph_id: str
    sentence_ids: list[str]
```

Functions:

```python
build_sentence_chunks(paper)
build_paragraph_chunks(paper)
build_equation_neighborhood_chunks(paper)
build_cross_reference_chunks(paper)
```

### Phase 2: Add Retrieval Layer

Add a `retrieval.py` module:

```python
class BM25Retriever:
    def fit(self, chunks): ...
    def search(self, query, top_k=10, filters=None): ...

class TfidfRetriever:
    def fit(self, chunks): ...
    def search(self, query, top_k=10, filters=None): ...

class HybridRetriever:
    def search(self, query, top_k=10): ...
```

The retriever should return:

```json
{
  "chunk_id": "...",
  "score": 8.42,
  "chunk_type": "equation_neighborhood",
  "text": "..."
}
```

### Phase 3: Rewrite Meaning/Symbol/Relation Extractors Around Evidence

Each extractor should take candidate chunks:

```python
meaning = meaning_extractor.extract(eq, retrieved_chunks, audit)
symbols = symbol_extractor.extract(eq, retrieved_chunks, audit)
relations = relation_extractor.extract(eq, all_equations, retrieved_chunks, audit)
```

The extractor should not search the whole paper blindly after retrieval unless
used as a fallback.

### Phase 4: Schema Validator

Add a final validation step:

Checks:

- every paper ID exists
- each equation has exactly `equation`, `meaning`, `symbols`, `relations`,
  `audit-trail`
- relations contain every other equation in the same paper
- every relation has `grade` and `description`
- grade is one of `none`, `potential`, `strong`
- no generated marker or empty audit trail for non-empty fields

## Report Argument

Suggested report wording:

> The system uses retrieval-augmented extraction, not retrieval-augmented
> generation. Scientific paper text is chunked into sentence, paragraph, and
> equation-neighborhood units. BM25 retrieves candidate evidence for each
> equation, symbol, and equation pair. Optional sentence embeddings are used
> only as a non-generative similarity signal. Final values are produced by
> deterministic extraction rules, gazetteer matches, and fixed-label
> classification. Therefore, every output value can be traced to arXiv text or
> to a documented deterministic decision rule.

## Risk Register

| Risk                                                              | Impact                        | Mitigation                                                  |
| ----------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------- |
| Retrieved chunk is relevant but does not contain exact definition | empty symbols or weak meaning | use top-k chunks and neighboring sentences                  |
| BM25 misses paraphrases                                           | lower recall                  | optional embedding rerank over BM25 candidates              |
| Dependency parser over-extracts definitions                       | false positives               | length filters and pattern priority                         |
| Relation descriptions look generated                              | compliance risk               | use fixed vocabulary labels only                            |
| PDF text extraction loses equation structure                      | poor LaTeX and context        | mark `pdf_text` in audit and keep quality discussion honest |
| Final relations do not include every pair                         | schema failure                | add validator before writing JSON                           |

## Best Final Recommendation

Use this as the main method:

```text
DOM-aware chunking
+ BM25 over paragraph/equation-neighborhood/sentence chunks
+ deterministic regex and spaCy extraction
+ fixed-label pairwise relation classifier
+ complete audit trail
```

Use this as optional experiment:

```text
BM25 top-k
+ all-MiniLM-L6-v2 cosine reranking
+ same deterministic final extraction
```

Avoid making embeddings the only method. BM25 is easier to explain, easier to
audit, and better aligned with the assignment's no-generation constraint.

## Sources Consulted

- Assignment PDF in this repository: `pra_nlp_s26.pdf`
- BM25/BM25F in Lucene paper: https://arxiv.org/abs/0911.5046
- BM25S paper: https://arxiv.org/abs/2407.03618
- SentenceTransformers semantic search docs: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
- `all-MiniLM-L6-v2` model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- spaCy rule-based matching docs: https://spacy.io/usage/rule-based-matching
- scikit-learn `TfidfVectorizer` docs: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- BeautifulSoup documentation: https://beautiful-soup-4.readthedocs.io/en/latest/
- PyMuPDF text extraction docs: https://pymupdf.readthedocs.io/en/latest/recipes-text.html
