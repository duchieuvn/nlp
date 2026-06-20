**Approach v2**

V2 is a retrieval-augmented extraction pipeline:

1. Structure each paper into sections, paragraphs, sentences, equations, and cross-references.
2. Create multiple evidence chunks around equations and symbols.
3. Use BM25 to retrieve relevant source passages.
4. Optionally rerank passages with sentence embeddings.
5. Extract meanings, symbol definitions, and relations using deterministic rules.
6. Assign fixed relation labels and preserve evidence in the audit trail.

Its key rule is: retrieval selects evidence, but never writes the answer.

**Comparison**

| Aspect       | Current `s4b.py`                                     | Approach v2                                                      |
| ------------ | ---------------------------------------------------- | ---------------------------------------------------------------- |
| Primary task | Equation meaning only                                | Meanings, symbols, and relations                                 |
| Search scope | Fixed local window before equation                   | Structured evidence across the paper                             |
| Candidates   | Short phrases containing predefined scientific terms | Sentences and spans retrieved from several chunk types           |
| Ranking      | MathBERT embedding similarity                        | BM25, optional embedding reranking, rule scores                  |
| Final answer | Highest-ranked candidate phrase                      | Deterministically extracted source sentence/span                 |
| Relations    | Not supported                                        | Pairwise classification using citations, symbols, and proximity  |
| Auditability | Candidate, score, and offsets                        | Chunk ID, source sentence, method, retrieval score, and evidence |
| Complexity   | Small and inexpensive                                | Larger indexing and extraction system                            |

The current method generates candidates using lexical rules ([s4b.py](/home/duchieuvn/Code/nlp/source/s4b.py:92)), embeds the context, formula, and candidates, then combines semantic similarity with proximity and cue bonuses ([s4b.py](/home/duchieuvn/Code/nlp/source/s4b.py:207)). It is simple, but only 186 of the current 353 equations produce candidates; 167 stop before MathBERT ranking.

V2 should improve recall because it can find definitions outside the immediate window and is substantially better suited to symbols and relations. It is also easier to defend under a strict non-generative requirement. Its disadvantages are greater implementation complexity, more tuning, and BM25’s weakness with paraphrases.

My recommendation is to use v2 as the main pipeline, with BM25 retrieval and deterministic extraction. Keep the current MathBERT similarity score as an optional reranking feature or experimental baseline, rather than the sole selection mechanism.
