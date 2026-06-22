"""Stage 3: Build retrieval chunks from structured documents."""


import json
from pathlib import Path

from config import CHUNKS_DIR, DOCUMENTS_DIR, OUTPUT_DIR

_CHUNK_TYPES = ("sentence", "paragraph", "section_context", "raw_equation",
                "raw_equation_neighborhood", "cross_reference")


def _sentence_chunks(doc: dict) -> list[dict]:
    chunks = []
    for sent in doc["sentences"].values():
        para = doc["paragraphs"].get(sent["paragraph_id"], {})
        sec_id = para.get("section_id", "")
        sec = next((s for s in doc["sections"] if s["section_id"] == sec_id), {})
        chunks.append({
            "chunk_id": f"{doc['paper_id']}:sentence:{sent['sentence_id']}",
            "chunk_type": "sentence",
            "paper_id": doc["paper_id"],
            "source_ids": {
                "sentence_id": sent["sentence_id"],
                "paragraph_id": sent["paragraph_id"],
                "section_id": sec_id,
            },
            "raw_equation_anchor": None,
            "document_order": para.get("document_order", 0),
            "section_title": sec.get("title", ""),
            "text": sent["text"],
            "visible_equation_labels": [],
        })
    return chunks


def _paragraph_chunks(doc: dict) -> list[dict]:
    chunks = []
    for para in doc["paragraphs"].values():
        sec_id = para["section_id"]
        sec = next((s for s in doc["sections"] if s["section_id"] == sec_id), {})
        chunks.append({
            "chunk_id": f"{doc['paper_id']}:paragraph:{para['paragraph_id']}",
            "chunk_type": "paragraph",
            "paper_id": doc["paper_id"],
            "source_ids": {
                "paragraph_id": para["paragraph_id"],
                "section_id": sec_id,
                "sentence_ids": para["sentence_ids"],
            },
            "raw_equation_anchor": None,
            "document_order": para["document_order"],
            "section_title": sec.get("title", ""),
            "text": para["text"],
            "visible_equation_labels": [],
        })
    return chunks


def _section_context_chunks(doc: dict) -> list[dict]:
    chunks = []
    for sec in doc["sections"]:
        if not sec["paragraph_ids"]:
            continue
        paras = [doc["paragraphs"][pid] for pid in sec["paragraph_ids"] if pid in doc["paragraphs"]]
        text = "\n\n".join(p["text"] for p in paras)
        if not text.strip():
            continue
        min_order = min((p["document_order"] for p in paras), default=0)
        chunks.append({
            "chunk_id": f"{doc['paper_id']}:section_context:{sec['section_id']}",
            "chunk_type": "section_context",
            "paper_id": doc["paper_id"],
            "source_ids": {
                "section_id": sec["section_id"],
                "paragraph_ids": sec["paragraph_ids"],
            },
            "raw_equation_anchor": None,
            "document_order": min_order,
            "section_title": sec.get("title", ""),
            "text": text,
            "visible_equation_labels": [],
        })
    return chunks


def _raw_equation_chunks(doc: dict) -> list[dict]:
    chunks = []
    for eq in doc["raw_equations"]:
        sec = next((s for s in doc["sections"] if s["section_id"] == eq["section_id"]), {})
        chunks.append({
            "chunk_id": f"{doc['paper_id']}:raw_equation:{eq['raw_equation_id']}",
            "chunk_type": "raw_equation",
            "paper_id": doc["paper_id"],
            "source_ids": {
                "raw_equation_id": eq["raw_equation_id"],
                "section_id": eq["section_id"],
            },
            "raw_equation_anchor": eq["anchor_id"],
            "document_order": eq["document_order"],
            "section_title": sec.get("title", ""),
            "text": eq["latex"],
            "visible_equation_labels": eq["visible_labels"],
        })
    return chunks


def _neighborhood_chunks(doc: dict) -> list[dict]:
    chunks = []
    sents = doc["sentences"]
    for eq in doc["raw_equations"]:
        sec = next((s for s in doc["sections"] if s["section_id"] == eq["section_id"]), {})
        before_texts = [sents[sid]["text"] for sid in eq["before_sentence_ids"] if sid in sents]
        after_texts = [sents[sid]["text"] for sid in eq["after_sentence_ids"] if sid in sents]
        parts = before_texts + [f"[EQUATION: {eq['latex']}]"] + after_texts
        text = " ".join(parts).strip()
        chunks.append({
            "chunk_id": f"{doc['paper_id']}:raw_equation_neighborhood:{eq['raw_equation_id']}",
            "chunk_type": "raw_equation_neighborhood",
            "paper_id": doc["paper_id"],
            "source_ids": {
                "raw_equation_id": eq["raw_equation_id"],
                "section_id": eq["section_id"],
                "before_sentence_ids": eq["before_sentence_ids"],
                "after_sentence_ids": eq["after_sentence_ids"],
            },
            "raw_equation_anchor": eq["anchor_id"],
            "document_order": eq["document_order"],
            "section_title": sec.get("title", ""),
            "text": text,
            "visible_equation_labels": eq["visible_labels"],
        })
    return chunks


def _cross_reference_chunks(doc: dict) -> list[dict]:
    chunks = []
    sents = doc["sentences"]
    for xref in doc["cross_references"]:
        sent = sents.get(xref["source_sentence_id"], {})
        sec = next((s for s in doc["sections"] if s["section_id"] == xref["source_section_id"]), {})
        para = doc["paragraphs"].get(xref["source_paragraph_id"], {})
        chunks.append({
            "chunk_id": f"{doc['paper_id']}:cross_reference:{xref['reference_id']}",
            "chunk_type": "cross_reference",
            "paper_id": doc["paper_id"],
            "source_ids": {
                "reference_id": xref["reference_id"],
                "section_id": xref["source_section_id"],
                "paragraph_id": xref["source_paragraph_id"],
                "sentence_id": xref["source_sentence_id"],
            },
            "raw_equation_anchor": None,
            "document_order": para.get("document_order", 0),
            "section_title": sec.get("title", ""),
            "text": sent.get("text", ""),
            "visible_equation_labels": xref["target_labels"],
        })
    return chunks


def _build_chunks(doc: dict) -> list[dict]:
    chunks = []
    chunks.extend(_sentence_chunks(doc))
    chunks.extend(_paragraph_chunks(doc))
    chunks.extend(_section_context_chunks(doc))
    chunks.extend(_raw_equation_chunks(doc))
    chunks.extend(_neighborhood_chunks(doc))
    chunks.extend(_cross_reference_chunks(doc))
    return chunks


def _write_chunks(paper_id: str, chunks: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"paper_id": paper_id, "chunk_count": len(chunks), "chunks": chunks}
    out = output_dir / f"{paper_id}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)


def run() -> dict:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_files = sorted(DOCUMENTS_DIR.glob("*.json"))
    if not doc_files:
        raise FileNotFoundError(f"No documents found in {DOCUMENTS_DIR}")

    results = []
    for doc_file in doc_files:
        doc = json.loads(doc_file.read_text(encoding="utf-8"))
        paper_id = doc["paper_id"]
        chunks = _build_chunks(doc)
        _write_chunks(paper_id, chunks, CHUNKS_DIR)
        by_type = {}
        for chunk in chunks:
            by_type[chunk["chunk_type"]] = by_type.get(chunk["chunk_type"], 0) + 1
        results.append({"paper_id": paper_id, "chunk_count": len(chunks), "by_type": by_type})

    report = {
        "paper_count": len(results),
        "total_chunks": sum(r["chunk_count"] for r in results),
        "papers": results,
    }
    return report
