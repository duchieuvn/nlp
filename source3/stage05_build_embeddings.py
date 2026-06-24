"""Stage 5: Build multi-vector MathBERT embedding spaces per paper."""


import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import (
    CHUNKS_DIR,
    CONTEXT_WINDOW,
    EMBEDDING_DIM,
    EMBEDDINGS_DIR,
    EQUATIONS_DIR,
    MATHBERT_MODEL,
    MAX_TOKENS,
    OUTPUT_DIR,
    SENTENCE_EQUATION_BUDGET,
    SUMMARY_EQUATION_BUDGET,
)

_EMBED_BATCH = 32  # max texts per forward pass — keeps GPU memory bounded

# ---------------------------------------------------------------------------
# Model loading (lazy, module-level singleton)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MATHBERT_MODEL)
    _model = AutoModel.from_pretrained(MATHBERT_MODEL)
    _model.eval()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)


# ---------------------------------------------------------------------------
# Token budget helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_budget(text: str, budget: int) -> tuple[str, bool]:
    if not text:
        return text, False
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text, False
    half = budget // 2
    head_ids = token_ids[:half]
    tail_ids = token_ids[-(budget - half):]
    head = _tokenizer.decode(head_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    tail = _tokenizer.decode(tail_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return head + " [...] " + tail, True


def _truncate_head(text: str, budget: int) -> tuple[str, bool]:
    if not text:
        return text, False
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text, False
    truncated = _tokenizer.decode(
        token_ids[:budget], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return truncated, True


def _truncate_tail(text: str, budget: int) -> tuple[str, bool]:
    if not text:
        return text, False
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text, False
    truncated = _tokenizer.decode(
        token_ids[-budget:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return truncated, True


# ---------------------------------------------------------------------------
# Input text builders
# ---------------------------------------------------------------------------

def _build_equation_text(latex: str) -> tuple[str, bool]:
    """equation: <latex> — head/tail truncation if over budget."""
    base = f"equation: {latex}"
    # MAX_TOKENS - 2 for [CLS]/[SEP]
    budget = MAX_TOKENS - 2
    text, truncated = _truncate_to_budget(base, budget)
    return text, truncated


def _build_summary_text(
    latex: str,
    before_texts: list[str],
    after_texts: list[str],
) -> tuple[str, bool, bool]:
    """Build a summary embedding input for an equation.

    Parameters
    ----------
    latex
        Equation LaTeX source.
    before_texts
        Sentences before the equation.
    after_texts
        Sentences after the equation.

    Returns
    -------
    tuple[str, bool, bool]
        Input text, whether the equation was truncated, and whether
        surrounding context was truncated.
    """
    special = 2  # [CLS] + [SEP]
    total_budget = MAX_TOKENS - special

    eq_part = f"equation: {latex}"
    eq_tokens = _tokenizer.encode(eq_part, add_special_tokens=False)
    eq_budget = min(SUMMARY_EQUATION_BUDGET, len(eq_tokens))
    if len(eq_tokens) > eq_budget:
        half = eq_budget // 2
        head = _tokenizer.decode(eq_tokens[:half], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tail = _tokenizer.decode(eq_tokens[-(eq_budget - half):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eq_part = f"equation: {head} [...] {tail}"
        eq_truncated = True
    else:
        eq_truncated = False

    context_budget = total_budget - eq_budget
    half_ctx = context_budget // 2

    before_text = " ".join(before_texts)
    after_text = " ".join(after_texts)
    before_ids = _tokenizer.encode(before_text, add_special_tokens=False)
    after_ids = _tokenizer.encode(after_text, add_special_tokens=False)

    # keep before tail, after head; redistribute unused
    before_used = min(len(before_ids), half_ctx)
    after_used = min(len(after_ids), half_ctx)
    leftover_before = half_ctx - before_used
    leftover_after = half_ctx - after_used
    after_used = min(len(after_ids), half_ctx + leftover_before)
    before_used = min(len(before_ids), half_ctx + leftover_after)

    before_final = _tokenizer.decode(
        before_ids[-before_used:] if before_used else [], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    after_final = _tokenizer.decode(
        after_ids[:after_used] if after_used else [], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    ctx_truncated = len(before_ids) > before_used or len(after_ids) > after_used

    parts = []
    if before_final.strip():
        parts.append(f"before: {before_final.strip()}")
    parts.append(eq_part)
    if after_final.strip():
        parts.append(f"after: {after_final.strip()}")
    return "\n".join(parts), eq_truncated, ctx_truncated


def _build_sentence_text(
    latex: str,
    sentence_text: str,
    position: str,
    distance: int,
) -> tuple[str, bool, bool]:
    """Build a sentence-conditioned equation embedding input.

    Parameters
    ----------
    latex
        Equation LaTeX source.
    sentence_text
        Context sentence paired with the equation.
    position
        Context position, usually ``before`` or ``after``.
    distance
        Sentence distance from the equation.

    Returns
    -------
    tuple[str, bool, bool]
        Input text, whether the equation was truncated, and whether the
        sentence context was truncated.
    """
    special = 2
    total_budget = MAX_TOKENS - special

    eq_part = f"equation: {latex}"
    eq_tokens = _tokenizer.encode(eq_part, add_special_tokens=False)
    eq_budget = min(SENTENCE_EQUATION_BUDGET, len(eq_tokens))

    sent_part = f"context: {sentence_text}\nposition: {position}\ndistance: {distance}"
    sent_tokens = _tokenizer.encode(sent_part, add_special_tokens=False)
    sent_budget = total_budget - eq_budget

    eq_truncated = False
    ctx_truncated = False

    if len(eq_tokens) > eq_budget:
        # try to give sentence its full space
        if len(sent_tokens) < sent_budget:
            extra = sent_budget - len(sent_tokens)
            eq_budget_actual = min(len(eq_tokens), eq_budget + extra)
        else:
            eq_budget_actual = eq_budget
        half = eq_budget_actual // 2
        head = _tokenizer.decode(eq_tokens[:half], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tail = _tokenizer.decode(eq_tokens[-(eq_budget_actual - half):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eq_part = f"equation: {head} [...] {tail}"
        eq_truncated = True

    if len(sent_tokens) > sent_budget:
        sent_part_truncated = _tokenizer.decode(
            sent_tokens[:sent_budget], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        sent_part = sent_part_truncated
        ctx_truncated = True

    return f"{eq_part}\n{sent_part}", eq_truncated, ctx_truncated


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize a list of texts with MathBERT.

    Parameters
    ----------
    texts
        Input strings to embed.

    Returns
    -------
    np.ndarray
        Matrix of normalized embeddings with shape ``(len(texts), 768)``.
    """
    _load_model()
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    batches: list[np.ndarray] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        encoding = _tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        encoding = {k: v.to(_device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = _model(**encoding)
        hidden = outputs.last_hidden_state  # (B, L, D)
        mask = encoding["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        batches.append(pooled / norms)
    return np.concatenate(batches, axis=0)


# ---------------------------------------------------------------------------
# Vector building for one equation
# ---------------------------------------------------------------------------

def _vectors_for_equation(eq: dict) -> list[dict]:
    """Build all equation-centric vector metadata rows.

    Parameters
    ----------
    eq
        Stage 4 equation record containing LaTeX and local context.

    Returns
    -------
    list[dict]
        Metadata rows for equation, summary, before-sentence, and
        after-sentence vectors.
    """
    paper_id = eq["paper_id"]
    equation_id = eq["equation_id"]
    latex = eq["latex"]
    before_sents = eq.get("before_sentences", [])
    after_sents = eq.get("after_sentences", [])

    rows: list[dict] = []

    # equation vector
    eq_text, eq_trunc = _build_equation_text(latex)
    rows.append({
        "vector_id": f"{paper_id}:equation:{equation_id}:equation",
        "equation_id": equation_id,
        "vector_kind": "equation",
        "chunk_id": None,
        "sentence_id": None,
        "context_position": None,
        "context_distance": 0,
        "source_text": eq_text,
        "input_token_count": _count_tokens(eq_text),
        "equation_truncated": eq_trunc,
        "context_truncated": False,
        "input_sha256": hashlib.sha256(eq_text.encode()).hexdigest(),
    })

    # summary vector
    before_texts = [s["text"] for s in before_sents]
    after_texts = [s["text"] for s in after_sents]
    summary_text, s_eq_trunc, s_ctx_trunc = _build_summary_text(latex, before_texts, after_texts)
    rows.append({
        "vector_id": f"{paper_id}:equation:{equation_id}:summary",
        "equation_id": equation_id,
        "vector_kind": "summary",
        "chunk_id": None,
        "sentence_id": None,
        "context_position": None,
        "context_distance": 0,
        "source_text": summary_text,
        "input_token_count": _count_tokens(summary_text),
        "equation_truncated": s_eq_trunc,
        "context_truncated": s_ctx_trunc,
        "input_sha256": hashlib.sha256(summary_text.encode()).hexdigest(),
    })

    # before_sentence vectors (nearest first = last in list)
    for dist, sent in enumerate(reversed(before_sents), start=1):
        sid = sent["sentence_id"]
        text, b_eq_trunc, b_ctx_trunc = _build_sentence_text(latex, sent["text"], "before", dist)
        rows.append({
            "vector_id": f"{paper_id}:equation:{equation_id}:before:{dist}",
            "equation_id": equation_id,
            "vector_kind": "before_sentence",
            "chunk_id": None,
            "sentence_id": sid,
            "context_position": "before",
            "context_distance": dist,
            "source_text": text,
            "input_token_count": _count_tokens(text),
            "equation_truncated": b_eq_trunc,
            "context_truncated": b_ctx_trunc,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })

    # after_sentence vectors (nearest first)
    for dist, sent in enumerate(after_sents, start=1):
        sid = sent["sentence_id"]
        text, a_eq_trunc, a_ctx_trunc = _build_sentence_text(latex, sent["text"], "after", dist)
        rows.append({
            "vector_id": f"{paper_id}:equation:{equation_id}:after:{dist}",
            "equation_id": equation_id,
            "vector_kind": "after_sentence",
            "chunk_id": None,
            "sentence_id": sid,
            "context_position": "after",
            "context_distance": dist,
            "source_text": text,
            "input_token_count": _count_tokens(text),
            "equation_truncated": a_eq_trunc,
            "context_truncated": a_ctx_trunc,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })

    return rows


# ---------------------------------------------------------------------------
# Family B — full-paper sentence chunk vectors
# ---------------------------------------------------------------------------

_CHUNK_EMBED_TYPES = frozenset({"sentence", "raw_equation_neighborhood"})


def _vectors_for_chunk_file(paper_id: str, chunk_file: Path) -> list[dict]:
    """Build full-paper chunk vector metadata rows.

    Parameters
    ----------
    paper_id
        arXiv paper identifier.
    chunk_file
        Path to the Stage 3 chunk JSON file.

    Returns
    -------
    list[dict]
        Metadata rows for embeddable sentence and equation-neighborhood
        chunks.
    """
    if not chunk_file.exists():
        return []
    data = json.loads(chunk_file.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for chunk in data.get("chunks", []):
        if chunk["chunk_type"] not in _CHUNK_EMBED_TYPES:
            continue
        text = chunk.get("text", "").strip()
        if not text:
            continue
        source_text, ctx_trunc = _truncate_head(text, MAX_TOKENS - 2)
        sent_id = chunk["source_ids"].get("sentence_id")
        rows.append({
            "vector_id": f"{paper_id}:sentence_chunk:{chunk['chunk_id']}",
            "equation_id": None,
            "vector_kind": "sentence_chunk",
            "chunk_id": chunk["chunk_id"],
            "sentence_id": sent_id,
            "context_position": None,
            "context_distance": 0,
            "source_text": source_text,
            "input_token_count": _count_tokens(source_text),
            "equation_truncated": False,
            "context_truncated": ctx_trunc,
            "input_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        })
    return rows


# ---------------------------------------------------------------------------
# Paper-level build
# ---------------------------------------------------------------------------

def _build_paper_space(
    paper_id: str,
    equations: list[dict],
    chunk_file: Path | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Build the complete embedding space for one paper.

    Parameters
    ----------
    paper_id
        arXiv paper identifier.
    equations
        Resolved Stage 4 equation records.
    chunk_file
        Optional Stage 3 chunk file used for full-paper retrieval rows.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        Embedding matrix and row metadata.
    """
    all_rows: list[dict] = []

    # Family A: equation-centric vectors
    for eq in sorted(equations, key=lambda e: e["equation_id"]):
        all_rows.extend(_vectors_for_equation(eq))

    # Family B: full-paper sentence chunk vectors
    if chunk_file is not None:
        all_rows.extend(_vectors_for_chunk_file(paper_id, chunk_file))

    texts = [r["source_text"] for r in all_rows]
    embeddings = _embed_texts(texts)  # (N, 768) — processed in batches of _EMBED_BATCH

    for idx, row in enumerate(all_rows):
        row["row"] = idx

    return embeddings, all_rows


def _write_paper_space(
    paper_id: str,
    embeddings: np.ndarray,
    rows: list[dict],
    output_dir: Path,
) -> None:
    """Write embedding arrays and metadata for one paper.

    Parameters
    ----------
    paper_id
        arXiv paper identifier.
    embeddings
        Normalized embedding matrix.
    rows
        Row metadata aligned with ``embeddings``.
    output_dir
        Directory where ``.npz`` and ``.json`` outputs are written.

    Returns
    -------
    None
        Files are written atomically to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{paper_id}.npz"
    npz_tmp = output_dir / f"{paper_id}.tmp.npz"
    np.savez(str(npz_tmp), embeddings=embeddings)
    npz_tmp.replace(npz_path)

    meta = {
        "paper_id": paper_id,
        "model": MATHBERT_MODEL,
        "pooling": "attention_mask_mean",
        "normalized": True,
        "dimension": EMBEDDING_DIM,
        "rows": rows,
    }
    json_path = output_dir / f"{paper_id}.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json_tmp.replace(json_path)


# ---------------------------------------------------------------------------
# Retrieval interface
# ---------------------------------------------------------------------------

def load_paper_space(paper_id: str) -> tuple[np.ndarray, list[dict]]:
    npz_path = EMBEDDINGS_DIR / f"{paper_id}.npz"
    json_path = EMBEDDINGS_DIR / f"{paper_id}.json"
    data = np.load(str(npz_path))
    embeddings = data["embeddings"].astype(np.float32)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return embeddings, meta["rows"]


def embed_query(query_text: str) -> np.ndarray:
    _load_model()
    return _embed_texts([query_text])[0]


def search_paper(
    paper_id: str,
    query_text: str,
    top_k: int,
    vector_kinds: list[str] | None = None,
    exclude_equation_ids: list[str] | None = None,
    group_by_equation: bool = False,
) -> list[dict[str, Any]]:
    """Search a paper embedding space by cosine similarity.

    Parameters
    ----------
    paper_id
        arXiv paper identifier.
    query_text
        Natural-language or math-aware retrieval query.
    top_k
        Maximum number of rows to return.
    vector_kinds
        Optional vector kinds to include.
    exclude_equation_ids
        Optional equation identifiers to exclude.
    group_by_equation
        If true, return only the best row for each equation.

    Returns
    -------
    list[dict[str, Any]]
        Ranked metadata rows with an added ``score`` field.
    """
    embeddings, rows = load_paper_space(paper_id)
    query_vec = embed_query(query_text).astype(np.float32)

    mask = np.ones(len(rows), dtype=bool)
    if vector_kinds:
        kinds_set = set(vector_kinds)
        mask &= np.array([r["vector_kind"] in kinds_set for r in rows])
    if exclude_equation_ids:
        excl = set(exclude_equation_ids)
        mask &= np.array([r["equation_id"] not in excl for r in rows])

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    sub_embeddings = embeddings[indices]
    scores = sub_embeddings @ query_vec  # cosine (already normalized)

    if group_by_equation:
        best: dict[str, tuple[float, int]] = {}
        for local_idx, global_idx in enumerate(indices):
            row = rows[global_idx]
            eq_id = row["equation_id"]
            score = float(scores[local_idx])
            if eq_id not in best or score > best[eq_id][0]:
                best[eq_id] = (score, global_idx)
        candidates = sorted(best.values(), key=lambda x: -x[0])[:top_k]
        return [
            {**rows[global_idx], "score": score}
            for score, global_idx in candidates
        ]

    top_local = np.argsort(-scores)[:top_k]
    return [
        {**rows[int(indices[i])], "score": float(scores[i])}
        for i in top_local
    ]


# ---------------------------------------------------------------------------
# Stage run
# ---------------------------------------------------------------------------

def run() -> dict:
    """Build embeddings for all Stage 4 equation files.

    Returns
    -------
    dict
        Build report containing model, device, vector counts, and
        per-paper statistics.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eq_files = sorted(EQUATIONS_DIR.glob("*.json"))
    if not eq_files:
        raise FileNotFoundError(f"No equation files found in {EQUATIONS_DIR}")

    _load_model()

    paper_results = []
    total_vectors = 0
    total_truncations = 0

    for eq_file in eq_files:
        payload = json.loads(eq_file.read_text(encoding="utf-8"))
        paper_id = payload["paper_id"]
        equations = [e for e in payload["equations"] if e.get("match_method") != "unresolved"]

        if not equations:
            continue

        chunk_file = CHUNKS_DIR / f"{paper_id}.json"
        embeddings, rows = _build_paper_space(paper_id, equations, chunk_file)
        _write_paper_space(paper_id, embeddings, rows, EMBEDDINGS_DIR)

        n_trunc = sum(1 for r in rows if r["equation_truncated"] or r["context_truncated"])
        n_eq_vecs = sum(1 for r in rows if r["vector_kind"] != "sentence_chunk")
        n_chunk_vecs = sum(1 for r in rows if r["vector_kind"] == "sentence_chunk")
        total_vectors += len(rows)
        total_truncations += n_trunc

        paper_results.append({
            "paper_id": paper_id,
            "equation_count": len(equations),
            "equation_vector_count": n_eq_vecs,
            "sentence_chunk_vector_count": n_chunk_vecs,
            "vector_count": len(rows),
            "truncation_count": n_trunc,
        })

    device_str = str(_device) if _device else "cpu"

    report = {
        "model": MATHBERT_MODEL,
        "device": device_str,
        "dimension": EMBEDDING_DIM,
        "paper_count": len(paper_results),
        "total_vectors": total_vectors,
        "total_truncations": total_truncations,
        "papers": paper_results,
    }
    return report
