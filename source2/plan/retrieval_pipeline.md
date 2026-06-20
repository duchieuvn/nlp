Use one generic retrieval layer. Meaning, symbol, and relation modules should only provide task-specific queries, filters, and extraction rules.

## Suggested Architecture

```text
source2/
├── retrieval/
│   ├── models.py          # Chunk, SearchQuery, SearchResult
│   ├── tokenizer.py       # Shared text/math tokenization
│   ├── filters.py         # Metadata filtering
│   ├── bm25.py            # Primary retriever
│   ├── tfidf.py           # Baseline retriever
│   ├── index_builder.py   # Load and index paper chunks
│   └── service.py         # Common search interface
│
├── meaning/
│   ├── query_builder.py
│   ├── extractor.py
│   └── scorer.py
│
├── symbol_definition/
│   ├── query_builder.py
│   ├── extractor.py
│   └── scorer.py
│
└── relation/
    ├── query_builder.py
    ├── evidence.py
    ├── classifier.py
    └── scorer.py
```

## Common Retrieval Interface

All tasks should create the same query object:

```python
@dataclass
class SearchQuery:
    text: str
    paper_id: str
    section_ids: list[str] | None = None
    chunk_types: list[str] | None = None
    equation_ids: list[str] | None = None
    symbols: list[str] | None = None
    top_k: int = 10
```

The retriever always returns:

```python
@dataclass
class SearchResult:
    chunk_id: str
    score: float
    chunk_type: str
    text: str
    section_id: str | None
    paragraph_ids: list[str]
    sentence_ids: list[str]
    nearby_equation_ids: list[str]
    symbols: list[str]
```

Usage:

```python
results = retrieval_service.search(query, method="bm25")
```

BM25 and TF-IDF implement the same interface, allowing easy comparison.

## Equation Meaning

Build a query from the equation ID and its symbols:

```text
Eq 3 Equation 3 defines describes represents gives Hamiltonian psi omega
```

Filters:

```python
chunk_types=[
    "equation_neighborhood",
    "paragraph",
    "sentence",
]
```

The meaning extractor then:

1. Searches for evidence.
2. Matches definition and description patterns.
3. Scores complete source sentences.
4. Selects the best sentence or returns an empty value.

Retrieval only finds candidates. `meaning/extractor.py` decides the meaning.

## Symbol Meaning

Create one query per symbol using its aliases:

```text
omega_c \omega_c ω_c where denotes represents is defined as
```

Filters:

```python
SearchQuery(
    paper_id=paper_id,
    symbols=["omega_c"],
    chunk_types=["sentence", "paragraph"],
)
```

The extractor applies rules such as:

```text
where omega_c denotes <definition>
omega_c is the <definition>
let omega_c be <definition>
```

The same retrieval index is reused; only the query template and extraction rules change.

## Equation Relations

For equations `N` and `M`, query:

```text
Eq N Eq M using substituting derived from equivalent reduces to
```

Search:

- Cross-reference chunks for explicit citations.
- Equation-neighborhood chunks for local derivation evidence.
- Paragraph and sentence chunks for relation cues.

Relation classification additionally uses non-retrieval features:

```python
RelationFeatures(
    explicit_reference=True,
    derivation_cue=True,
    shared_symbols=["H", "psi"],
    same_section=True,
    context_similarity=0.72,
)
```

A deterministic classifier converts these features into:

```json
{
  "grade": "strong",
  "description": "derived from"
}
```

## Reuse Boundary

The retrieval layer owns:

- Loading and indexing chunks
- Tokenization
- Metadata filtering
- BM25 and TF-IDF scoring
- Returning ranked evidence

Task modules own:

- Query construction
- Extraction patterns
- Candidate rescoring
- Confidence thresholds
- Final values and audit records

This boundary keeps retrieval reusable and prevents it from accidentally producing answers.
