# Project Description

This project builds a prototype system for extracting an equation knowledge graph from quantum physics research papers on arXiv. The goal is not to summarize whole papers, but to collect structured information about important numbered equations and the relationships between them.

The input is a paper list file of the form `paper_list_<exam ID>.txt`. The papers must be processed strictly in the given order, and only the papers from that list may be used. For every paper, the system should inspect the arXiv source material and extract only enumerated equations, usually the equations marked with numbers such as `(1)`, `(2)`, and so on.

The dataset must be written as JSON. The top-level keys are arXiv IDs. Under each paper key, the value is an equation dictionary whose keys are the equation numbers as they appear in the paper. Every paper must be present in the dataset, even if it contains no enumerated equations, in which case the nested equation dictionary stays empty.

For each paper, the prototype only needs to keep the first 7 relevant enumerated equations. If a paper contains fewer than 7 such equations, all of them are kept. The full dataset should contain between 350 and 356 equations overall. Processing must continue paper by paper until the lower bound is reached, but the last paper being processed must always be included completely even if the cut-off is crossed inside that paper.

Each extracted equation entry must contain these fields:

- `equation`: the equation in LaTeX form
- `meaning`: a short description of what the equation expresses, or the name of the equation when applicable
- `symbols`: a dictionary that explains relevant abbreviations and symbols used in the equation
- `relations`: a dictionary that links this equation to all other relevant equations in the same paper
- `audit-trail`: a compact log of the extraction methods that produced the final fields

The `symbols` dictionary should explain paper-specific notation such as variable names, abbreviations, and special operators when they are introduced or defined in the surrounding text. Standard mathematical operators do not need explanation. The `relations` dictionary should be graph-ready and compare every equation against the others in the same paper. Relation grades must be one of `none`, `potential`, or `strong`, where `strong` is reserved for direct, clearly justified relations such as equivalence, derivation, generalization, special case, or explicit reference. `potential` is used when a relation seems plausible but is not fully established from the local evidence.

The audit trail is a required part of the prototype because it demonstrates how each result was obtained and makes the extraction process inspectable. The audit trail should show which implemented methods fired, what text or structure they found, and how those findings were converted into meanings, symbol definitions, and relations. It should be short, precise, and useful for debugging or quality review.

All information extraction must be based only on arXiv sources such as HTML, PDF, or LaTeX. No prompting is allowed, and no external APIs or language-model services may be used for text generation. This constraint means the system should rely on deterministic parsing, pattern matching, structural cues, and evidence from nearby paper text rather than generated interpretations.

The documentation for the project should focus on three things: the key ideas behind the extraction approach, how well the method generalizes to arbitrary quantum physics papers, and a critical discussion of quality. That discussion should include what the prototype is good at, where it is likely to fail, and which limitations cannot easily be removed because they are inherent to the source material or the no-prompt restriction.

The pipeline must also respect arXiv crawling rules and use conservative request pacing so that paper retrieval does not overload the server.
