"""Stage 8: Extract symbol definitions from top-K retrieved sentence chunks."""

import json
import re

import numpy as np

import stage05_build_embeddings as _emb
from config import CHUNKS_DIR, EMBEDDINGS_DIR, EQUATIONS_DIR, OUTPUT_DIR, SYMBOL_MEANINGS_DIR, SYMBOLS_DIR

_TOP_K = 30
_MAX_DEF_WORDS = 20
_MIN_DEF_WORDS = 2
_STOP_WORDS = frozenset("""
    a an the this that these those is are was were be been being have has had do
    does did will would could should may might shall can and or but for at by in
    on to of up as if it its we they them their with from not all any both each
    few more most other some such than then too very
""".split())


def _meaningful_words(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[A-Za-z]+", text)
        if w.lower() not in _STOP_WORDS
    ]


def _valid_span(span: str, equation_latex: str) -> bool:
    span = span.strip()
    if not span:
        return False
    words = span.split()
    if len(words) > _MAX_DEF_WORDS:
        return False
    if len(_meaningful_words(span)) < _MIN_DEF_WORDS:
        return False
    if "\\begin{" in span:
        return False
    if equation_latex and span in equation_latex:
        return False
    latex_cmds = re.findall(r"\\[A-Za-z]+", span)
    if latex_cmds and len(latex_cmds) > max(len(words), 1) * 0.5:
        return False
    return True


def _alias_patterns(symbol: dict) -> list[str]:
    aliases = [symbol.get("canonical", ""), symbol.get("unicode", ""), symbol.get("base", "")]
    aliases.extend(symbol.get("latex_forms", []))
    patterns = []
    seen = set()
    for alias in aliases:
        if not alias:
            continue
        escaped = re.escape(alias)
        if alias.startswith("\\"):
            escaped = escaped.replace(r"\{", r"\{?").replace(r"\}", r"\}?")
            pat = escaped + r"(?![A-Za-z])"
        elif re.fullmatch(r"[A-Za-z0-9_]+", alias):
            pat = r"\b" + escaped + r"\b"
        else:
            pat = escaped
        if pat not in seen:
            seen.add(pat)
            patterns.append(pat)
    return patterns


def _build_sym_regex(symbol: dict) -> re.Pattern | None:
    patterns = _alias_patterns(symbol)
    if not patterns:
        return None
    try:
        return re.compile("|".join(patterns))
    except re.error:
        return None


def _span_after(text: str, start: int) -> str:
    rest = text[start:].strip(" ,;:.")
    stop = re.search(r"[.;]|\s+where\s+|\s+and\s+(?=\\|[A-Za-z_]+\s+(?:is|are|denotes?))", rest)
    return (rest[:stop.start()] if stop else rest).strip(" ,;:.")


def _try_where_is(text: str, sym_pat: re.Pattern, equation_latex: str) -> str | None:
    m = re.search(r"\bwhere\s+(?:" + sym_pat.pattern + r")\s+(?:is|are)\s+", text, re.I)
    if not m:
        return None
    span = _span_after(text, m.end())
    return span if _valid_span(span, equation_latex) else None


def _try_sym_denotes(text: str, sym_pat: re.Pattern, equation_latex: str) -> str | None:
    m = re.search(r"(?:^|(?<=\s)|(?<=\())(?:" + sym_pat.pattern + r")\s+(?:denotes?|represents?)\s+", text, re.I)
    if not m:
        return None
    span = _span_after(text, m.end())
    return span if _valid_span(span, equation_latex) else None


def _try_let_be(text: str, sym_pat: re.Pattern, equation_latex: str) -> str | None:
    m = re.search(r"\blet\s+(?:" + sym_pat.pattern + r")\s+(?:be|denote)\s+", text, re.I)
    if not m:
        return None
    span = _span_after(text, m.end())
    return span if _valid_span(span, equation_latex) else None


def _try_reverse_np(text: str, sym_pat: re.Pattern, equation_latex: str) -> str | None:
    m = re.search(r"\b((?:the\s+)?(?:[A-Za-z][A-Za-z-]*\s+){1,5})(?:" + sym_pat.pattern + r")(?=\s|[,.;)]|$)", text, re.I)
    if not m:
        return None
    span = m.group(1).strip()
    return span if _valid_span(span, equation_latex) else None


def _try_respectively(text: str, sym_pat: re.Pattern, all_pats: list[re.Pattern], equation_latex: str) -> str | None:
    if "respectively" not in text.lower():
        return None
    m = sym_pat.search(text)
    if not m:
        return None
    resp = re.search(r"\brespectively\b", text, re.I)
    if not resp or m.end() > resp.start():
        return None
    positions = []
    for pat in all_pats:
        hit = pat.search(text[:resp.start()])
        if hit:
            positions.append(hit.start())
    positions = sorted(set(positions))
    try:
        idx = positions.index(m.start())
    except ValueError:
        idx = 0
    rhs = re.search(r"\bare\s+(?:the\s+)?(.*?)\s*,?\s*respectively", text, re.I)
    if not rhs:
        return None
    defs = [d.strip(" ,") for d in re.split(r"\s+and\s+|,\s*", rhs.group(1)) if d.strip(" ,")]
    if idx < len(defs) and _valid_span(defs[idx], equation_latex):
        return defs[idx]
    return None


def _load_sentence_chunk_index(paper_id: str, chunks: dict) -> tuple[np.ndarray, list[dict], dict[str, dict]]:
    embeddings, rows = _emb.load_paper_space(paper_id)
    indices = [i for i, r in enumerate(rows) if r.get("vector_kind") == "sentence_chunk"]
    matrix = embeddings[indices].astype(np.float32)
    selected_rows = [rows[i] for i in indices]
    chunk_lookup = {c["chunk_id"]: c for c in chunks.get("chunks", [])}
    return matrix, selected_rows, chunk_lookup


def _retrieved_candidates(query: str, matrix: np.ndarray, rows: list[dict], chunk_lookup: dict[str, dict]) -> list[dict]:
    if matrix.size == 0:
        return []
    query_vec = _emb.embed_query(query).astype(np.float32)
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:_TOP_K]
    candidates = []
    for rank, local_idx in enumerate(order, start=1):
        row = rows[int(local_idx)]
        chunk = chunk_lookup.get(row.get("chunk_id"), {})
        text = chunk.get("text") or row.get("source_text", "")
        candidates.append({
            "text": text,
            "sentence_id": row.get("sentence_id") or chunk.get("source_ids", {}).get("sentence_id") or row.get("chunk_id"),
            "retrieval_rank": rank,
        })
    return candidates


def _query_for_symbol(symbol: dict) -> str:
    aliases = [symbol.get("canonical", ""), symbol.get("unicode", ""), symbol.get("base", "")]
    aliases.extend(symbol.get("latex_forms", []))
    aliases = [a for a in dict.fromkeys(aliases) if a]
    seed = aliases[0] if aliases else symbol.get("canonical", "")
    return " ".join([f"where {seed} is", f"{seed} denotes", *aliases])


def _find_definition(symbol: dict, candidates: list[dict], all_pats: list[re.Pattern], equation_latex: str) -> dict | None:
    sym_pat = _build_sym_regex(symbol)
    if sym_pat is None:
        return None
    extractors = [
        ("where_is", 1.0, lambda t: _try_where_is(t, sym_pat, equation_latex)),
        ("denotes", 0.9, lambda t: _try_sym_denotes(t, sym_pat, equation_latex)),
        ("let_be", 0.8, lambda t: _try_let_be(t, sym_pat, equation_latex)),
        ("reverse_np", 0.65, lambda t: _try_reverse_np(t, sym_pat, equation_latex)),
        ("respectively", 0.75, lambda t: _try_respectively(t, sym_pat, all_pats, equation_latex)),
    ]
    for candidate in candidates:
        for pattern, confidence, extractor in extractors:
            span = extractor(candidate["text"])
            if span:
                return {
                    "definition": span,
                    "sentence_id": candidate["sentence_id"],
                    "pattern": pattern,
                    "confidence": confidence,
                    "retrieval_rank": candidate["retrieval_rank"],
                }
    return None


def _process_paper(paper_id: str) -> dict:
    sym_data = json.loads((SYMBOLS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    chunk_data = json.loads((CHUNKS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    eq_data = json.loads((EQUATIONS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    eq_latex = {e["equation_id"]: e.get("latex", "") for e in eq_data.get("equations", [])}
    matrix, rows, chunk_lookup = _load_sentence_chunk_index(paper_id, chunk_data)

    results = []
    for eq in sym_data.get("equations", []):
        symbols = eq.get("symbols", [])
        all_pats = [p for s in symbols if (p := _build_sym_regex(s))]
        meanings = {}
        for symbol in symbols:
            query = _query_for_symbol(symbol)
            candidates = _retrieved_candidates(query, matrix, rows, chunk_lookup)
            found = _find_definition(symbol, candidates, all_pats, eq_latex.get(eq["equation_id"], ""))
            if found:
                meanings[symbol["canonical"]] = found
        results.append({"equation_id": eq["equation_id"], "symbol_meanings": meanings})
    return {"paper_id": paper_id, "equations": results}


def run() -> dict:
    SYMBOL_MEANINGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sym_files = sorted(SYMBOLS_DIR.glob("*.json"))
    if not sym_files:
        raise FileNotFoundError(f"No symbol files in {SYMBOLS_DIR}")

    total_defs = 0
    paper_results = []
    for sym_file in sym_files:
        paper_id = sym_file.stem
        if not (EMBEDDINGS_DIR / f"{paper_id}.json").exists():
            result = {"paper_id": paper_id, "equations": []}
        else:
            result = _process_paper(paper_id)
        out = SYMBOL_MEANINGS_DIR / f"{paper_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)

        defs = sum(len(e.get("symbol_meanings", {})) for e in result["equations"])
        total_defs += defs
        paper_results.append({"paper_id": paper_id, "definitions_found": defs})

    return {
        "paper_count": len(sym_files),
        "total_definitions_found": total_defs,
        "papers": paper_results,
    }
