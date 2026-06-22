"""Stage 4: Match reviewed equations to raw HTML equations in structured documents."""


import json
import re
from pathlib import Path

from config import (
    ANNOTATIONS_FILE,
    CONTEXT_WINDOW,
    DOCUMENTS_DIR,
    EQUATION_ALIGNMENT_REPORT,
    EQUATIONS_DIR,
    EQUATIONS_FILE,
    OUTPUT_DIR,
)

_WHITESPACE = re.compile(r"\s+")


def _normalize_latex(latex: str) -> str:
    latex = latex.replace("%\n", "").replace("\n", " ")
    latex = re.sub(r"^\s*\\displaystyle\s*", "", latex)
    latex = re.sub(r"\\begin\{(?:aligned|split)\}", "", latex)
    latex = re.sub(r"\\end\{(?:aligned|split)\}", "", latex)
    latex = _WHITESPACE.sub("", latex)
    return latex.rstrip(".,")


def _annotation_dom_ids(annotation_html_list: list[str]) -> list[str]:
    from bs4 import BeautifulSoup
    ids: list[str] = []
    for html in annotation_html_list:
        ann = BeautifulSoup(html, "html.parser").find("annotation")
        if ann and ann.get("id"):
            ids.append(ann["id"])
    return list(dict.fromkeys(ids))


def _audit_anchor(entry: dict) -> str | None:
    for item in reversed(entry.get("audit-trail", [])):
        if isinstance(item, dict):
            ctx = item.get("context_extraction")
            if isinstance(ctx, dict):
                anchor = ctx.get("anchor_id")
                if isinstance(anchor, str):
                    return anchor
                matched = ctx.get("matched_annotation_id")
                if isinstance(matched, str):
                    return matched
    return None


def _overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return sum(1 for c in shorter if c in longer) / max(len(longer), 1)


class _Matcher:
    def __init__(self, raw_equations: list[dict]) -> None:
        self._raws = raw_equations
        self._by_anchor: dict[str, list[dict]] = {}
        self._by_label: dict[str, list[dict]] = {}
        self._by_norm_latex: dict[str, list[dict]] = {}
        for eq in raw_equations:
            anchor = eq.get("anchor_id")
            if anchor:
                self._by_anchor.setdefault(anchor, []).append(eq)
            for label in eq.get("visible_labels", []):
                self._by_label.setdefault(label, []).append(eq)
            norm = _normalize_latex(eq.get("latex", ""))
            if norm:
                self._by_norm_latex.setdefault(norm, []).append(eq)

    def match(
        self,
        equation_id: str,
        entry: dict,
        annotation_html_list: list[str],
    ) -> tuple[dict | None, str]:
        # Strategy 1: annotation DOM IDs
        ann_ids = _annotation_dom_ids(annotation_html_list)
        for dom_id in ann_ids:
            candidates = self._by_anchor.get(dom_id, [])
            if len(candidates) == 1:
                return candidates[0], "annotation_dom_id"
            if len(candidates) > 1:
                # prefer exact anchor match
                exact = [c for c in candidates if c.get("anchor_id") == dom_id]
                if len(exact) == 1:
                    return exact[0], "annotation_dom_id"

        # Strategy 2: audit anchor
        audit_anchor = _audit_anchor(entry)
        if audit_anchor:
            candidates = self._by_anchor.get(audit_anchor, [])
            if len(candidates) == 1:
                return candidates[0], "audit_anchor"

        # Strategy 3: unique visible label
        label_candidates = self._by_label.get(equation_id, [])
        if len(label_candidates) == 1:
            return label_candidates[0], "visible_label"

        # Strategy 4: visible label + normalized LaTeX
        if len(label_candidates) > 1:
            norm = _normalize_latex(entry.get("equation", ""))
            if norm:
                matches = [c for c in label_candidates if _normalize_latex(c.get("latex", "")) == norm]
                if len(matches) == 1:
                    return matches[0], "visible_label_and_latex"

        # Strategy 5: exact normalized LaTeX
        norm = _normalize_latex(entry.get("equation", ""))
        if norm:
            exact = self._by_norm_latex.get(norm, [])
            if len(exact) == 1:
                return exact[0], "exact_latex"

        # Strategy 6: unique high-overlap normalized LaTeX (≥ 0.9)
        if norm:
            high_overlap = [
                eq for eq in self._raws
                if _overlap_ratio(norm, _normalize_latex(eq.get("latex", ""))) >= 0.9
            ]
            if len(high_overlap) == 1:
                return high_overlap[0], "high_overlap_latex"

        return None, "unresolved"


def _build_equation_output(
    paper_id: str,
    equation_id: str,
    entry: dict,
    raw_eq: dict | None,
    match_method: str,
    sentences: dict,
) -> dict:
    if raw_eq is not None:
        before_ids = raw_eq["before_sentence_ids"][:CONTEXT_WINDOW]
        after_ids = raw_eq["after_sentence_ids"][:CONTEXT_WINDOW]
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
    else:
        before_sents = []
        after_sents = []

    return {
        "paper_id": paper_id,
        "equation_id": equation_id,
        "latex": entry.get("equation", ""),
        "raw_equation_id": raw_eq["raw_equation_id"] if raw_eq else None,
        "anchor_id": raw_eq["anchor_id"] if raw_eq else None,
        "section_id": raw_eq["section_id"] if raw_eq else None,
        "document_order": raw_eq["document_order"] if raw_eq else None,
        "match_method": match_method,
        "before_sentences": before_sents,
        "after_sentences": after_sents,
        "audit": _audit_context(entry),
    }


def _audit_context(entry: dict) -> dict:
    for item in reversed(entry.get("audit-trail", [])):
        if isinstance(item, dict) and isinstance(item.get("context_extraction"), dict):
            return item["context_extraction"]
    return {}


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

    all_equations = json.loads(EQUATIONS_FILE.read_text(encoding="utf-8"))
    all_annotations = json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))

    paper_ids = [pid for pid, entries in all_equations.items() if entries]

    alignment_records: list[dict] = []
    paper_results: list[dict] = []

    for paper_id in paper_ids:
        doc_file = DOCUMENTS_DIR / f"{paper_id}.json"
        if not doc_file.exists():
            raise FileNotFoundError(f"Missing document for {paper_id}: {doc_file}")

        doc = json.loads(doc_file.read_text(encoding="utf-8"))
        sentences = doc["sentences"]
        raw_equations = doc["raw_equations"]
        matcher = _Matcher(raw_equations)

        reviewed = all_equations[paper_id]
        annotations = all_annotations.get(paper_id, {})

        output_equations: list[dict] = []
        paper_resolved = 0
        paper_unresolved = 0

        for equation_id, entry in reviewed.items():
            ann_list = annotations.get(equation_id, [])
            raw_eq, method = matcher.match(equation_id, entry, ann_list)

            if method == "unresolved":
                paper_unresolved += 1
            else:
                paper_resolved += 1

            eq_out = _build_equation_output(
                paper_id, equation_id, entry, raw_eq, method, sentences
            )
            output_equations.append(eq_out)
            alignment_records.append({
                "paper_id": paper_id,
                "equation_id": equation_id,
                "match_method": method,
                "raw_equation_id": raw_eq["raw_equation_id"] if raw_eq else None,
            })

        _write_paper_equations(paper_id, output_equations, EQUATIONS_DIR)
        paper_results.append({
            "paper_id": paper_id,
            "equation_count": len(output_equations),
            "resolved": paper_resolved,
            "unresolved": paper_unresolved,
        })

    methods: dict[str, int] = {}
    for rec in alignment_records:
        methods[rec["match_method"]] = methods.get(rec["match_method"], 0) + 1

    report = {
        "paper_count": len(paper_ids),
        "total_equations": sum(r["equation_count"] for r in paper_results),
        "total_resolved": sum(r["resolved"] for r in paper_results),
        "total_unresolved": sum(r["unresolved"] for r in paper_results),
        "by_method": methods,
        "papers": paper_results,
        "alignment": alignment_records,
    }

    tmp = EQUATION_ALIGNMENT_REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(EQUATION_ALIGNMENT_REPORT)

    return report
