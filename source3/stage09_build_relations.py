"""Stage 9: Build directed relation pairs between reviewed equations in each paper."""

import json
import re

import numpy as np

from config import (
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    EQUATIONS_DIR,
    OUTPUT_DIR,
    RELATION_SEMANTIC_THRESHOLD,
    RELATIONS_DIR,
    SYMBOLS_DIR,
)

_DERIVATION_CUES = re.compile(
    r"\b(?:from|using|by|follows?\s+from|derived?\s+(?:from|using)|"
    r"applying|substitut(?:ing|e)|plugging|combining)\b",
    re.IGNORECASE,
)
_EQUIVALENCE_CUES = re.compile(
    r"\b(?:equivalent\s+to|same\s+as|reduces?\s+to|simplif(?:ies|y)\s+to|"
    r"equals?|identical\s+to|equal\s+to|recovers?)\b",
    re.IGNORECASE,
)
_SPECIAL_CASE_CUES = re.compile(
    r"\b(?:special\s+case|particular\s+case|setting\s+\S+\s+(?:=|to)|"
    r"when\s+\S+\s+=|limit(?:ing\s+case)?|reducing\s+to)\b",
    re.IGNORECASE,
)
_EQ_LABEL_REF = re.compile(
    r"(?:eq(?:uation|n)?s?\.?\s*)?\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)", re.IGNORECASE
)


def _label_from_anchor(anchor_id: str | None) -> str | None:
    if not anchor_id:
        return None
    m = re.search(r"\.E(\d+(?:\.\d+)?)$", anchor_id)
    return m.group(1) if m else None


def _labels_mentioned(text: str) -> set[str]:
    return {m.group(1) for m in _EQ_LABEL_REF.finditer(text)}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _section_proximity(sec_a: str | None, sec_b: str | None) -> float:
    if not sec_a or not sec_b:
        return 0.0
    if sec_a == sec_b:
        return 1.0
    # Adjacent sections share same parent prefix
    parts_a = sec_a.split(":")
    parts_b = sec_b.split(":")
    if parts_a[:-1] == parts_b[:-1]:
        return 0.5
    return 0.0


def _process_paper(paper_id: str) -> dict:
    """Build directed relation records for one paper.

    Parameters
    ----------
    paper_id
        arXiv identifier whose equations, chunks, symbols, and optional
        embeddings should be combined.

    Returns
    -------
    dict
        Stage 9 payload containing graph-ready ordered equation pairs
        with relation grade and description.
    """
    eq_data = json.loads((EQUATIONS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    chunk_data = json.loads((CHUNKS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))

    sym_file = SYMBOLS_DIR / f"{paper_id}.json"
    sym_data = json.loads(sym_file.read_text(encoding="utf-8")) if sym_file.exists() else {"equations": []}

    emb_json_path = EMBEDDINGS_DIR / f"{paper_id}.json"
    if emb_json_path.exists():
        emb_meta = json.loads(emb_json_path.read_text(encoding="utf-8"))
        embeddings = np.load(str(EMBEDDINGS_DIR / f"{paper_id}.npz"))["embeddings"].astype(np.float32)
    else:
        emb_meta = {"rows": []}
        embeddings = np.zeros((0, 768), dtype=np.float32)

    # Resolved equations only
    equations = [e for e in eq_data["equations"] if e.get("match_method") != "unresolved"]
    if len(equations) < 2:
        return {"paper_id": paper_id, "relations": {}}

    # Summary vector index: equation_id -> embedding row index
    summary_row: dict[str, int] = {
        row["equation_id"]: row["row"]
        for row in emb_meta["rows"]
        if row["vector_kind"] == "summary"
    }

    # Symbol canonical sets per equation
    sym_by_eq: dict[str, set[str]] = {}
    for eq_sym in sym_data["equations"]:
        sym_by_eq[eq_sym["equation_id"]] = {s["canonical"] for s in eq_sym["symbols"]}

    # Label→equation_id mapping
    label_to_eq: dict[str, str] = {}
    for eq in equations:
        label = _label_from_anchor(eq.get("anchor_id"))
        if label:
            label_to_eq[label] = eq["equation_id"]

    # All neighborhood sentences per equation (before + after)
    neighborhood_texts: dict[str, list[str]] = {}
    for eq in equations:
        neighborhood_texts[eq["equation_id"]] = [
            s["text"] for s in eq.get("before_sentences", []) + eq.get("after_sentences", [])
        ]

    # Cross-reference chunks for citation detection
    xref_chunks = [c for c in chunk_data["chunks"] if c["chunk_type"] == "cross_reference"]

    def _detect_cue(texts: list[str], b_label: str | None) -> str | None:
        """Return the first matching cue type for B mentioned in texts."""
        for text in texts:
            if b_label and b_label not in _labels_mentioned(text):
                continue
            if _DERIVATION_CUES.search(text):
                return "derivation"
            if _EQUIVALENCE_CUES.search(text):
                return "equivalence"
            if _SPECIAL_CASE_CUES.search(text):
                return "special_case"
            if b_label:
                return "citation"
        return None

    relations: dict[str, dict[str, dict]] = {}

    for eq_a in equations:
        eid_a = eq_a["equation_id"]
        label_a = _label_from_anchor(eq_a.get("anchor_id"))
        doc_order_a = eq_a.get("document_order") or 0
        relations[eid_a] = {}

        for eq_b in equations:
            eid_b = eq_b["equation_id"]
            if eid_a == eid_b:
                continue

            label_b = _label_from_anchor(eq_b.get("anchor_id")) or eid_b

            # --- Feature 1: explicit cue in A's neighborhood texts ---
            nbhd_texts_a = neighborhood_texts.get(eid_a, [])
            cue = _detect_cue(nbhd_texts_a, label_b)

            # --- Feature 2: cross-reference chunks near A that cite B ---
            if not cue:
                near_xrefs = [
                    c for c in xref_chunks
                    if label_b in c.get("visible_equation_labels", [])
                    and abs((c.get("document_order") or 0) - doc_order_a) < 200
                ]
                if near_xrefs:
                    cue = "citation"

            # --- Feature 3: shared canonical symbols ---
            sym_a = sym_by_eq.get(eid_a, set())
            sym_b = sym_by_eq.get(eid_b, set())
            jaccard = _jaccard(sym_a, sym_b)

            # --- Feature 4: section proximity ---
            sec_prox = _section_proximity(eq_a.get("section_id"), eq_b.get("section_id"))

            # --- Feature 5: semantic similarity via summary vectors ---
            sem_sim = 0.0
            row_a = summary_row.get(eid_a)
            row_b = summary_row.get(eid_b)
            if row_a is not None and row_b is not None:
                sem_sim = float(np.dot(embeddings[row_a], embeddings[row_b]))

            # --- Grade and description ---
            if cue:
                grade = "strong"
                desc_map = {
                    "citation": "explicit citation",
                    "derivation": "derived from",
                    "equivalence": "equivalent",
                    "special_case": "special case",
                }
                description = desc_map[cue]
            elif jaccard >= 0.4:
                grade = "potential"
                description = "shares symbols"
            elif sem_sim >= RELATION_SEMANTIC_THRESHOLD:
                grade = "potential"
                description = "same section context"
            elif sec_prox > 0:
                grade = "potential"
                description = "same section context"
            else:
                grade = "none"
                description = "none"

            relations[eid_a][eid_b] = {"grade": grade, "description": description}

    return {"paper_id": paper_id, "relations": relations}


def run() -> dict:
    """Run Stage 9 over all selected-equation files.

    Returns
    -------
    dict
        Summary counts for processed papers, relation pairs, strong
        relations, and potential relations.

    Raises
    ------
    FileNotFoundError
        If no Stage 4 equation files are available.
    """
    RELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eq_files = sorted(EQUATIONS_DIR.glob("*.json"))
    if not eq_files:
        raise FileNotFoundError(f"No equation files in {EQUATIONS_DIR}")

    total_pairs = 0
    total_strong = 0
    total_potential = 0
    paper_results = []

    for eq_file in eq_files:
        paper_id = eq_file.stem
        result = _process_paper(paper_id)

        out = RELATIONS_DIR / f"{paper_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)

        pairs = sum(len(v) for v in result["relations"].values())
        strong = sum(
            1 for v in result["relations"].values()
            for r in v.values() if r["grade"] == "strong"
        )
        potential = sum(
            1 for v in result["relations"].values()
            for r in v.values() if r["grade"] == "potential"
        )
        total_pairs += pairs
        total_strong += strong
        total_potential += potential
        paper_results.append({
            "paper_id": paper_id,
            "pairs": pairs,
            "strong": strong,
            "potential": potential,
        })

    return {
        "paper_count": len(eq_files),
        "total_pairs": total_pairs,
        "total_strong": total_strong,
        "total_potential": total_potential,
        "papers": paper_results,
    }
