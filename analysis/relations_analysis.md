# Relation Analysis for `data/final_data.json`

This analysis summarizes the relation entries currently stored in
`data/final_data.json`.

## Overall Coverage

- Paper keys in final export: **56**
- Papers with at least one equation: **54**
- Empty paper dictionaries: **2**
- Total equations: **356**
- Directed relation entries: **2080**
- Expected directed pairs within non-empty papers: **2080**
- Missing directed pairs: **0**

The relation graph is therefore complete at the directed-pair level: for
each paper, every selected equation has a relation entry to every other
selected equation in the same paper.

## Grade Distribution

| Grade | Count | Percentage |
|---|---:|---:|
| potential | 2000 | 96.15% |
| strong | 80 | 3.85% |

Most relations are weak/potential rather than strong. This is expected
from the current Stage 9 design, because explicit equation-to-equation
statements are relatively rare in papers, while fallback signals such as
same-section context fire often.

## Description Distribution

| Description | Count | Percentage |
|---|---:|---:|
| same section context | 1864 | 89.62% |
| shares symbols | 136 | 6.54% |
| explicit citation | 48 | 2.31% |
| derived from | 26 | 1.25% |
| special case | 6 | 0.29% |

The dominant relation type is `same section context`. This means that
the graph is dense, but many edges represent weak contextual association
rather than a precise mathematical dependency.

## Strong Relations

There are **80** strong relations:

- **48 explicit citations**
- **26 derived-from relations**
- **6 special-case relations**

Strong relations are produced when Stage 9 finds a direct equation
reference together with a cue phrase, or a direct citation without a more
specific cue. These are the most reliable relation entries because they
are supported by textual evidence in the paper.

## Potential Relations

There are **2000** potential relations:

- **1864 same-section-context relations**
- **136 shared-symbol relations**

Potential relations should be interpreted as graph candidates, not as
confirmed derivations. They are useful for producing a connected
graph-ready structure, but they are less precise than explicit
relations.

## Directionality

The output stores directed relations. However, many potential relations
are symmetric because they come from symmetric signals such as section
proximity or shared symbols.

- Reciprocal unordered pairs with the same grade and description: **966**
- Reciprocal unordered pairs with different grade or description: **74**
- One-way missing reverse edges: **0**

This means the graph is complete in both directions, but only a small
number of equation pairs have asymmetric evidence.

## Papers Dominated by Same-Section Relations

The following papers have all relation entries described as
`same section context`:

- `2403.03204`
- `2501.04264`
- `2401.02625`
- `2408.00252`
- `2502.09920`
- `2412.07479`
- `2508.11712`
- `2410.17702`
- `2411.04765`
- `2404.07802`
- `2408.07132`
- `2411.02913`
- `2411.08214`
- `2503.09459`
- `2405.02630`

These cases show the main weakness of the relation stage: if several
equations appear in the same section but the paper does not explicitly
state dependencies between them, the current fallback connects all pairs
as potential relations.

## Quality Interpretation

The relation extraction is strong for auditability and coverage:

- Every possible directed pair is represented.
- Strong relations are tied to explicit textual cues.
- The output is graph-ready because every equation has relation entries
  to the other equations in the same paper.

The main limitation is precision for potential relations:

- Same-section context is a broad fallback and can over-connect equations.
- Dense sections with many equations produce many weak edges.
- A `potential` edge should not be read as a confirmed derivation,
  equivalence, or dependency.

For the report, the relation results should be described as a two-level
system: strong edges are evidence-backed relation claims, while potential
edges are graph candidates based on weaker contextual or symbolic
similarity.

## Possible Improvement

A stricter version of Stage 9 could reduce over-connection by requiring
same-section relations to satisfy at least one additional signal, such as
shared symbols or high MathBERT cosine similarity. This would reduce the
number of weak edges and make the graph sparser, but it may also remove
some useful candidate relations when papers do not explicitly describe
their equation dependencies.
