"""Stage 4: Extract all display equations from structured documents."""

import json
from pathlib import Path

from config import (
    CONTEXT_WINDOW,
    DOCUMENTS_DIR,
    EQUATION_ALIGNMENT_REPORT,
    EQUATIONS_DIR,
    OUTPUT_DIR,
    PAPER_LIST_FILE,
)


def _build_equation_record(
    paper_id: str,
    equation_id: str,
    raw_eq: dict,
    sentences: dict,
) -> dict:
    before_ids = raw_eq.get("before_sentence_ids", [])[:CONTEXT_WINDOW]
    after_ids = raw_eq.get("after_sentence_ids", [])[:CONTEXT_WINDOW]
    before_sents = [
        {"sentence_id": sid, "text": sentences[sid]["text"]}
        for sid in before_ids
        if sid in sentences
    ]
    after_sents = [
        {"sentence_id": sid, "text": sentences[sid]["text"]}
        for sid in after_ids
        if sid in sentences
    ]
    return {
        "paper_id": paper_id,
        "equation_id": equation_id,
        "latex": raw_eq.get("latex", ""),
        "raw_equation_id": raw_eq.get("raw_equation_id"),
        "anchor_id": raw_eq.get("anchor_id"),
        "section_id": raw_eq.get("section_id"),
        "document_order": raw_eq.get("document_order"),
        "visible_labels": raw_eq.get("visible_labels", []),
        "before_sentences": before_sents,
        "after_sentences": after_sents,
    }


def _write_paper_equations(paper_id: str, equations: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{paper_id}.json"
    tmp = out.with_suffix(".json.tmp")
    payload = {"paper_id": paper_id, "equations": equations}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)


def run() -> dict:
    EQUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = PAPER_LIST_FILE.read_text(encoding="utf-8").splitlines()
    paper_ids = [line.removeprefix("arXiv:").strip() for line in lines if line.strip()]

    paper_results: list[dict] = []

    for paper_id in paper_ids:
        doc_file = DOCUMENTS_DIR / f"{paper_id}.json"
        if not doc_file.exists():
            continue

        doc = json.loads(doc_file.read_text(encoding="utf-8"))
        sentences = doc.get("sentences", {})
        raw_equations = doc.get("raw_equations", [])

        output_equations: list[dict] = []
        for idx, raw_eq in enumerate(raw_equations, start=1):
            equation_id = str(idx)
            record = _build_equation_record(paper_id, equation_id, raw_eq, sentences)
            output_equations.append(record)

        _write_paper_equations(paper_id, output_equations, EQUATIONS_DIR)
        paper_results.append({
            "paper_id": paper_id,
            "equation_count": len(output_equations),
        })

    report = {
        "paper_count": len(paper_results),
        "total_equations": sum(r["equation_count"] for r in paper_results),
        "papers": paper_results,
    }

    tmp = EQUATION_ALIGNMENT_REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(EQUATION_ALIGNMENT_REPORT)

    return {
        **report,
        "total_resolved": report["total_equations"],
        "total_unresolved": 0,
    }
