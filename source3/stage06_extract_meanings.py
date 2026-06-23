"""Stage 6: Extract equation meanings as verbatim sentence spans."""

import json
import math
import re
from collections import Counter

import numpy as np

from config import CHUNKS_DIR, EMBEDDINGS_DIR, EQUATIONS_DIR, MEANINGS_DIR, OUTPUT_DIR

_CUE_PATTERNS = [
    re.compile(r"\b(?:Eq\.?|Equation)\s*\(?\s*{label}\s*\)?\s+(?:states|gives|defines|describes|represents|is)\b", re.I),
    re.compile(r"\(\s*{label}\s*\)\s+(?:gives|states|defines|describes|represents|is)\b", re.I),
    re.compile(r"\b(?:is\s+defined\s+as|is\s+given\s+by|represents|describes|denotes)\b", re.I),
]
_PROCEDURAL = re.compile(r"^\s*(?:We|To|Note\s+that|It\s+follows)\b")
_SYMBOL_DEFINITION = re.compile(
    r"\b(?:where\s+\\?[A-Za-z][A-Za-z0-9_{}^\\]*\s+(?:is|are)|"
    r"\\?[A-Za-z][A-Za-z0-9_{}^\\]*\s+denotes?|let\s+\\?[A-Za-z])\b",
    re.I,
)
_TOKEN = re.compile(r"[A-Za-z]+|\\[A-Za-z]+|[0-9]+")
_BROAD_RETRIEVAL_TOP_K = 20


def _label_from_anchor(anchor_id: str | None) -> str | None:
    if not anchor_id:
        return None
    m = re.search(r"\.E(\d+(?:\.\d+)?)$", anchor_id)
    return m.group(1) if m else None


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def _latex_density(text: str) -> float:
    words = max(_word_count(text), 1)
    latexish = len(re.findall(r"(?:\\[A-Za-z]+|[_^{}=+\-*/]|[A-Za-z]_[A-Za-z0-9])", text))
    return latexish / words


def _incomplete(text: str) -> bool:
    stripped = text.strip()
    return (
        _word_count(stripped) < 5
        or stripped.endswith((",", ":"))
        or stripped.count("(") > stripped.count(")")
    )


def _hard_filter_reasons(text: str) -> list[str]:
    reasons = []
    if _PROCEDURAL.match(text):
        reasons.append("procedural")
    if _latex_density(text) > 0.4:
        reasons.append("math_heavy")
    if _SYMBOL_DEFINITION.search(text):
        reasons.append("symbol_definition")
    if _incomplete(text):
        reasons.append("incomplete")
    return reasons


def _hard_filtered(text: str) -> bool:
    return bool(_hard_filter_reasons(text))


def _tokenize(text: str) -> list[str]:
    return [t.lower().lstrip("\\") for t in _TOKEN.findall(text)]


def _bm25_scores(query: str, candidates: list[dict]) -> dict[int, float]:
    query_terms = _tokenize(query)
    if not query_terms or not candidates:
        return {i: 0.0 for i in range(len(candidates))}

    docs = [_tokenize(c["text"]) for c in candidates]
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    df = Counter(term for doc in docs for term in set(doc))
    q_counts = Counter(query_terms)
    scores: dict[int, float] = {}
    k1 = 1.2
    b = 0.75
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        dl = len(doc) or 1
        score = 0.0
        for term, qf in q_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (tf[term] * (k1 + 1) / denom) * qf
        scores[i] = score
    max_score = max(scores.values(), default=0.0)
    if max_score > 0:
        scores = {i: s / max_score for i, s in scores.items()}
    return scores


def _bm25_rank(query: str, chunks: list[dict], top_k: int) -> list[tuple[float, dict]]:
    """Rank chunks by BM25-style lexical overlap with a query."""
    if not chunks:
        return []
    pseudo_candidates = [{"text": c.get("text", "")} for c in chunks]
    scores = _bm25_scores(query, pseudo_candidates)
    ranked = [
        (score, chunks[i])
        for i, score in scores.items()
        if score > 0 and chunks[i].get("text", "").strip()
    ]
    ranked.sort(key=lambda item: (-item[0], abs(item[1].get("document_order") or 0)))
    return ranked[:top_k]


def _load_embedding_indexes(paper_id: str) -> tuple[np.ndarray | None, dict, dict]:
    meta_path = EMBEDDINGS_DIR / f"{paper_id}.json"
    npz_path = EMBEDDINGS_DIR / f"{paper_id}.npz"
    if not meta_path.exists() or not npz_path.exists():
        return None, {}, {}
    rows = json.loads(meta_path.read_text(encoding="utf-8")).get("rows", [])
    embeddings = np.load(str(npz_path))["embeddings"].astype(np.float32)
    summary_rows = {
        r["equation_id"]: r["row"]
        for r in rows
        if r.get("vector_kind") == "summary" and r.get("equation_id")
    }
    sentence_rows: dict[str, int] = {}
    chunk_rows: dict[str, int] = {}
    for r in rows:
        if r.get("vector_kind") != "sentence_chunk":
            continue
        if r.get("sentence_id"):
            sentence_rows[r["sentence_id"]] = r["row"]
        if r.get("chunk_id"):
            chunk_rows[r["chunk_id"]] = r["row"]
    return embeddings, summary_rows, {"sentence": sentence_rows, "chunk": chunk_rows}


def _source_bonus(source: str) -> float:
    return {
        "cross_reference": 1.0,
        "cue_pattern": 0.7,
        "bm25_retrieval": 0.55,
        "proximity": 0.4,
    }.get(source, 0.0)


def _candidate_vector_row(candidate: dict, row_indexes: dict) -> int | None:
    sid = candidate.get("sentence_id")
    if sid and sid in row_indexes.get("sentence", {}):
        return row_indexes["sentence"][sid]
    cid = candidate.get("chunk_id")
    if cid and cid in row_indexes.get("chunk", {}):
        return row_indexes["chunk"][cid]
    return None


def _process_paper(paper_id: str) -> dict:
    eq_data = json.loads((EQUATIONS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    chunk_data = json.loads((CHUNKS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    embeddings, summary_rows, row_indexes = _load_embedding_indexes(paper_id)

    chunks = chunk_data.get("chunks", [])
    xrefs = [c for c in chunks if c.get("chunk_type") == "cross_reference"]
    neighborhoods = [c for c in chunks if c.get("chunk_type") == "raw_equation_neighborhood"]
    sentence_chunks = [
        c for c in chunks
        if c.get("chunk_type") in {"sentence", "raw_equation_neighborhood"}
    ]

    results = []
    for eq in eq_data.get("equations", []):
        equation_id = eq["equation_id"]
        if eq.get("match_method") == "unresolved":
            results.append({
                "equation_id": equation_id,
                "meaning": "",
                "source_sentence_id": None,
                "score": 0.0,
                "match_source": "none",
            })
            continue

        anchor = eq.get("anchor_id")
        label = _label_from_anchor(anchor)
        doc_order = eq.get("document_order") or 0
        candidates: list[dict] = []
        seen: set[str] = set()

        def add_candidate(
            text: str,
            source: str,
            sentence_id=None,
            chunk_id=None,
            order=None,
            retrieval_score: float = 0.0,
        ) -> None:
            text = re.sub(r"\s+", " ", (text or "").strip())
            key = text.casefold()
            if not text or key in seen:
                return
            seen.add(key)
            candidates.append({
                "text": text,
                "source": source,
                "sentence_id": sentence_id,
                "chunk_id": chunk_id,
                "document_order": order if order is not None else doc_order,
                "retrieval_score": retrieval_score,
            })

        if label:
            for chunk in xrefs:
                if label in chunk.get("visible_equation_labels", []):
                    add_candidate(
                        chunk.get("text", ""),
                        "cross_reference",
                        chunk.get("source_ids", {}).get("sentence_id"),
                        chunk.get("chunk_id"),
                        chunk.get("document_order"),
                    )

        cue_regexes = [
            re.compile(p.pattern.format(label=re.escape(label or equation_id)), p.flags)
            for p in _CUE_PATTERNS
        ]
        before = eq.get("before_sentences", [])
        for sent in before:
            text = sent.get("text", "")
            if any(p.search(text) for p in cue_regexes):
                add_candidate(text, "cue_pattern", sent.get("sentence_id"), None, doc_order)

        for chunk in neighborhoods:
            if chunk.get("raw_equation_anchor") != anchor:
                continue
            text = chunk.get("text", "")
            if any(p.search(text) for p in cue_regexes):
                add_candidate(
                    text,
                    "cue_pattern",
                    chunk.get("source_ids", {}).get("sentence_id"),
                    chunk.get("chunk_id"),
                    chunk.get("document_order"),
                )

        section_title = ""
        for sent in before:
            if sent.get("text"):
                break
        retrieval_query = " ".join([
            eq.get("latex", ""),
            f"Equation {label or equation_id}",
            section_title,
            " ".join(sent.get("text", "") for sent in before[-2:]),
        ])
        for bm25_score, chunk in _bm25_rank(retrieval_query, sentence_chunks, _BROAD_RETRIEVAL_TOP_K):
            add_candidate(
                chunk.get("text", ""),
                "bm25_retrieval",
                chunk.get("source_ids", {}).get("sentence_id"),
                chunk.get("chunk_id"),
                chunk.get("document_order"),
                bm25_score,
            )

        if not candidates and before:
            nearest_before = before[-1]
            add_candidate(nearest_before.get("text", ""), "proximity", nearest_before.get("sentence_id"), None, doc_order)

        if not candidates:
            results.append({
                "equation_id": equation_id,
                "meaning": "",
                "source_sentence_id": None,
                "score": 0.0,
                "match_source": "none",
                "source_text": "",
                "audit": {
                    "candidate_count": 0,
                    "selection_method": "no_candidates",
                },
            })
            continue

        bm25 = _bm25_scores(eq.get("latex", ""), candidates)
        summary_row = summary_rows.get(equation_id)
        scored = []
        for i, c in enumerate(candidates):
            cosine = 0.0
            cand_row = _candidate_vector_row(c, row_indexes)
            if embeddings is not None and summary_row is not None and cand_row is not None:
                cosine = float(np.dot(embeddings[summary_row], embeddings[cand_row]))
                cosine = (cosine + 1.0) / 2.0
            distance = abs((c.get("document_order") or doc_order) - doc_order)
            proximity = 1.0 / (1.0 + distance)
            filter_reasons = _hard_filter_reasons(c["text"])
            filter_penalty = min(0.25, 0.08 * len(filter_reasons))
            score = (
                0.5 * cosine
                + 0.2 * bm25.get(i, 0.0)
                + 0.2 * _source_bonus(c["source"])
                + 0.1 * proximity
                - filter_penalty
            )
            scored.append((score, c, filter_reasons))

        best_score, best, best_filter_reasons = max(
            scored,
            key=lambda item: (
                item[0],
                -len(item[2]),
                item[1].get("retrieval_score", 0.0),
                item[1]["text"],
            ),
        )
        results.append({
            "equation_id": equation_id,
            "meaning": best["text"],
            "source_text": best["text"],
            "source_sentence_id": best.get("sentence_id"),
            "score": round(float(best_score), 4),
            "match_source": best["source"],
            "selection_method": "combined_score_with_audited_filters",
            "audit": {
                "candidate_count": len(candidates),
                "hard_filtered": bool(best_filter_reasons),
                "hard_filter_reasons": best_filter_reasons,
                "candidate_source_counts": dict(Counter(c["source"] for c in candidates)),
            },
        })

    return {"paper_id": paper_id, "equations": results}


def run() -> dict:
    MEANINGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eq_files = sorted(EQUATIONS_DIR.glob("*.json"))
    if not eq_files:
        raise FileNotFoundError(f"No equation files in {EQUATIONS_DIR}")

    total_found = 0
    total_empty = 0
    paper_results = []
    for eq_file in eq_files:
        paper_id = eq_file.stem
        result = _process_paper(paper_id)
        out = MEANINGS_DIR / f"{paper_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)

        found = sum(1 for e in result["equations"] if e["meaning"])
        empty = sum(1 for e in result["equations"] if not e["meaning"])
        total_found += found
        total_empty += empty
        paper_results.append({
            "paper_id": paper_id,
            "equations": len(result["equations"]),
            "meanings_found": found,
            "meanings_empty": empty,
        })

    return {
        "paper_count": len(eq_files),
        "total_meanings_found": total_found,
        "total_meanings_empty": total_empty,
        "papers": paper_results,
    }
