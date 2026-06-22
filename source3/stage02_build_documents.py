"""Stage 2: Parse HTML into structured documents with raw equations and context."""


import json
import re
from bisect import bisect_left, bisect_right, insort
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from config import DOCUMENTS_DIR, EQUATIONS_FILE, HTML_DIR, OUTPUT_DIR

SCHEMA_VERSION = "2.0"

_WHITESPACE = re.compile(r"\s+")
_SECTION_CLASSES: dict[str, tuple[int, str]] = {
    "ltx_section": (1, "section"),
    "ltx_subsection": (2, "subsection"),
    "ltx_subsubsection": (3, "subsubsection"),
    "ltx_paragraph": (4, "paragraph"),
    "ltx_appendix": (1, "appendix"),
}
_EXCLUDED_SECTION_CLASSES = {"ltx_bibliography"}
_EXCLUDED_ANCESTOR_TAGS = {"figure", "table", "nav", "footer", "header"}
_EQUATION_CLASSES = {"ltx_equation", "ltx_equationgroup"}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _normalize_latex(latex: str) -> str:
    latex = latex.replace("%\n", "").replace("\n", " ")
    latex = re.sub(r"^\s*\\displaystyle\s*", "", latex)
    latex = re.sub(r"\\begin\{(?:aligned|split)\}", "", latex)
    latex = re.sub(r"\\end\{(?:aligned|split)\}", "", latex)
    latex = _WHITESPACE.sub("", latex)
    return latex.rstrip(".,")


def _clean_text_node(node) -> str:
    node = copy(node)
    for math in node.find_all("math"):
        alttext = math.get("alttext")
        math.replace_with(f" {alttext} " if alttext else " ")
    for tag in node.find_all(["annotation", "annotation-xml", "semantics", "script", "style"]):
        tag.decompose()
    return _normalize_text(node.get_text(" ", strip=True))


# ---------------------------------------------------------------------------
# Sentence segmentation (spaCy sentencizer)
# ---------------------------------------------------------------------------

class _Segmenter:
    def __init__(self) -> None:
        import spacy
        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")

    def spans(self, text: str) -> list[tuple[str, int, int]]:
        if not text:
            return []
        doc = self._nlp(text)
        return [
            (sent.text, sent.start_char, sent.end_char)
            for sent in doc.sents
            if sent.text.strip()
        ]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SentenceRecord:
    sentence_id: str
    paragraph_id: str
    order: int
    text: str
    start: int
    end: int
    cross_reference_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParagraphRecord:
    paragraph_id: str
    section_id: str
    document_order: int
    text: str
    sentence_ids: list[str] = field(default_factory=list)
    nearby_raw_equation_ids: list[str] = field(default_factory=list)
    cross_reference_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectionRecord:
    section_id: str
    parent_section_id: str | None
    order: int
    level: int
    kind: str
    title: str
    synthetic: bool
    paragraph_ids: list[str] = field(default_factory=list)
    raw_equation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RawEquationRecord:
    raw_equation_id: str
    anchor_id: str | None
    visible_labels: list[str]
    latex: str
    document_order: int
    section_id: str | None
    before_sentence_ids: list[str] = field(default_factory=list)
    after_sentence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrossReferenceRecord:
    reference_id: str
    raw_text: str
    reference_type: str
    source_section_id: str
    source_paragraph_id: str
    source_sentence_id: str
    sentence_start: int
    sentence_end: int
    target_labels: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cross-reference extraction
# ---------------------------------------------------------------------------

_XREF_PATTERN = re.compile(
    r"\b(?P<prefix>Eqs?\.|Equations?)\s*"
    r"(?P<body>\([^)]{1,80}\)(?:\s*[-–—]\s*\([^)]{1,40}\))?|"
    r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)",
    re.IGNORECASE,
)
_LABEL_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")
_RANGE_PATTERN = re.compile(r"^\(?\s*(\d+)\s*\)?\s*[-–—]\s*\(?\s*(\d+)\s*\)?$")


def _parse_reference_labels(body: str) -> tuple[str, list[str]]:
    m = _RANGE_PATTERN.match(body.strip())
    if m:
        start, end = map(int, m.groups())
        if start <= end <= start + 100:
            return "range", [str(v) for v in range(start, end + 1)]
    labels = _LABEL_PATTERN.findall(body)
    return ("list" if len(labels) > 1 else "singular"), labels


def _extract_cross_references(
    paper_id: str,
    sections: list[SectionRecord],
    paragraphs: dict[str, ParagraphRecord],
    sentences: dict[str, SentenceRecord],
) -> list[CrossReferenceRecord]:
    refs: list[CrossReferenceRecord] = []
    for section in sections:
        for para_id in section.paragraph_ids:
            para = paragraphs[para_id]
            for sent_id in para.sentence_ids:
                sent = sentences[sent_id]
                for match in _XREF_PATTERN.finditer(sent.text):
                    ref_type, labels = _parse_reference_labels(match.group("body"))
                    if not labels:
                        continue
                    ref_id = f"{paper_id}:xref:{len(refs) + 1}"
                    ref = CrossReferenceRecord(
                        reference_id=ref_id,
                        raw_text=match.group(0),
                        reference_type=ref_type,
                        source_section_id=section.section_id,
                        source_paragraph_id=para_id,
                        source_sentence_id=sent_id,
                        sentence_start=match.start(),
                        sentence_end=match.end(),
                        target_labels=labels,
                    )
                    refs.append(ref)
                    para.cross_reference_ids.append(ref_id)
                    sent.cross_reference_ids.append(ref_id)
    return refs


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _classes(node) -> set[str]:
    return set(node.get("class") or [])


def _section_spec(node) -> tuple[int, str] | None:
    for cls, spec in _SECTION_CLASSES.items():
        if cls in _classes(node):
            return spec
    return None


def _heading_text(section_node) -> str:
    h = section_node.find(re.compile(r"^h[1-6]$"), recursive=False) or \
        section_node.find(re.compile(r"^h[1-6]$"))
    return _normalize_text(h.get_text(" ", strip=True)) if h else ""


def _document_title(soup: BeautifulSoup) -> str:
    for selector in [
        lambda: soup.find(class_="ltx_title_document"),
        lambda: soup.find("h1"),
        lambda: soup.title,
    ]:
        node = selector()
        if node:
            return _normalize_text(node.get_text(" ", strip=True))
    return ""


def _is_excluded_paragraph(node) -> bool:
    for ancestor in node.parents:
        tag = getattr(ancestor, "name", None)
        if tag in _EXCLUDED_ANCESTOR_TAGS:
            return True
        if tag and _EXCLUDED_SECTION_CLASSES & _classes(ancestor):
            return True
    return False


def _is_equation_table(node) -> bool:
    return (
        node
        and node.name == "table"
        and bool(_EQUATION_CLASSES & _classes(node))
    )


def _equation_table_for(node):
    if _is_equation_table(node):
        return node
    return node.find_parent(_is_equation_table) if node else None


def _visible_labels(table) -> list[str]:
    labels = []
    for node in table.find_all(
        class_=lambda c: c and ("ltx_eqn_eqno" in c if isinstance(c, list) else "ltx_eqn_eqno" in c)
    ):
        label = re.sub(r"\s+", "", node.get_text(" ", strip=True)).strip("()")
        if label:
            labels.append(label)
    return list(dict.fromkeys(labels))


def _table_latex(table) -> str:
    parts = []
    for ann in table.find_all("annotation"):
        if ann.get("encoding") == "application/x-tex":
            raw = ann.get_text()
            if raw.strip():
                parts.append(raw.strip())
    if parts:
        return " ".join(parts)
    for math in table.find_all("math"):
        alt = math.get("alttext")
        if alt:
            return alt
    return ""


def _parse_html(
    paper_id: str,
    html_text: str,
    html_source: str,
    segmenter: _Segmenter,
    context_window: int = 5,
) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    title = _document_title(soup)

    # --- assign position counter to all nodes ---
    position: dict[int, int] = {}
    counter = 0
    for node in soup.descendants:
        if hasattr(node, "name") and node.name:
            position[id(node)] = counter
            counter += 1

    # --- collect sections ---
    sections: list[SectionRecord] = []
    section_for_node: dict[int, str] = {}
    section_stack: list[tuple[int, str]] = []
    section_count: dict[str, int] = {}

    def _make_section_id(kind: str) -> str:
        n = section_count.get(kind, 0) + 1
        section_count[kind] = n
        return f"sec:{kind}:{n}"

    # synthetic preamble section
    preamble = SectionRecord(
        section_id="sec:preamble",
        parent_section_id=None,
        order=0,
        level=0,
        kind="preamble",
        title="",
        synthetic=True,
    )
    sections.append(preamble)

    for node in soup.find_all(True):
        spec = _section_spec(node)
        if spec is None:
            continue
        if _EXCLUDED_SECTION_CLASSES & _classes(node):
            continue
        level, kind = spec
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()
        parent_id = section_stack[-1][1] if section_stack else None
        sec_id = _make_section_id(kind)
        sec = SectionRecord(
            section_id=sec_id,
            parent_section_id=parent_id,
            order=len(sections),
            level=level,
            kind=kind,
            title=_heading_text(node),
            synthetic=False,
        )
        sections.append(sec)
        section_stack.append((level, sec_id))
        section_for_node[id(node)] = sec_id

    def _section_id_for(node) -> str:
        for ancestor in [node] + list(node.parents):
            sid = section_for_node.get(id(ancestor))
            if sid:
                return sid
        return "sec:preamble"

    # --- collect paragraphs and sentences ---
    paragraphs: dict[str, ParagraphRecord] = {}
    sentences: dict[str, SentenceRecord] = {}
    para_doc_orders: list[int] = []

    para_count: dict[str, int] = {}

    def _make_para_id(sec_id: str) -> str:
        n = para_count.get(sec_id, 0) + 1
        para_count[sec_id] = n
        return f"{sec_id}:p{n}"

    for p_node in soup.find_all("p", class_=lambda c: c and "ltx_p" in c):
        if _is_excluded_paragraph(p_node):
            continue
        text = _clean_text_node(p_node)
        if not text:
            continue
        sec_id = _section_id_for(p_node)
        doc_order = position.get(id(p_node), len(paragraphs))
        para_id = _make_para_id(sec_id)
        sents = segmenter.spans(text)
        sent_ids: list[str] = []
        for idx, (s_text, s_start, s_end) in enumerate(sents, start=1):
            sent_id = f"{para_id}:s{idx}"
            sentences[sent_id] = SentenceRecord(
                sentence_id=sent_id,
                paragraph_id=para_id,
                order=idx,
                text=s_text,
                start=s_start,
                end=s_end,
            )
            sent_ids.append(sent_id)
        para = ParagraphRecord(
            paragraph_id=para_id,
            section_id=sec_id,
            document_order=doc_order,
            text=text,
            sentence_ids=sent_ids,
        )
        paragraphs[para_id] = para
        for sec in sections:
            if sec.section_id == sec_id:
                sec.paragraph_ids.append(para_id)
                break
        insort(para_doc_orders, doc_order)

    # sorted paragraph list by document_order
    sorted_paras = sorted(paragraphs.values(), key=lambda p: p.document_order)
    sorted_para_orders = [p.document_order for p in sorted_paras]

    # --- collect raw equations ---
    raw_equations: list[RawEquationRecord] = []
    eq_index = 0

    for table in soup.find_all("table"):
        if not _is_equation_table(table):
            continue
        doc_order = position.get(id(table), 0)
        sec_id = _section_id_for(table)
        anchor = table.get("id") or None
        labels = _visible_labels(table)
        latex = _table_latex(table)
        eq_index += 1
        raw_eq_id = f"raw:{eq_index:04d}"

        # find up to context_window prose sentences before and after
        # by locating adjacent paragraphs by document_order
        prev_idx = bisect_left(sorted_para_orders, doc_order) - 1
        next_idx = bisect_right(sorted_para_orders, doc_order)

        before_sent_ids: list[str] = []
        for offset in range(context_window):
            idx = prev_idx - offset
            if idx < 0:
                break
            para = sorted_paras[idx]
            for sent_id in reversed(para.sentence_ids):
                before_sent_ids.append(sent_id)
                if len(before_sent_ids) >= context_window:
                    break
            if len(before_sent_ids) >= context_window:
                break
        before_sent_ids = list(reversed(before_sent_ids))

        after_sent_ids: list[str] = []
        for offset in range(context_window):
            idx = next_idx + offset
            if idx >= len(sorted_paras):
                break
            para = sorted_paras[idx]
            for sent_id in para.sentence_ids:
                after_sent_ids.append(sent_id)
                if len(after_sent_ids) >= context_window:
                    break
            if len(after_sent_ids) >= context_window:
                break

        raw_eq = RawEquationRecord(
            raw_equation_id=raw_eq_id,
            anchor_id=anchor,
            visible_labels=labels,
            latex=latex,
            document_order=doc_order,
            section_id=sec_id,
            before_sentence_ids=before_sent_ids[:context_window],
            after_sentence_ids=after_sent_ids[:context_window],
        )
        raw_equations.append(raw_eq)

        # attach to section
        for sec in sections:
            if sec.section_id == sec_id:
                sec.raw_equation_ids.append(raw_eq_id)
                break

        # mark nearby paragraphs
        if prev_idx >= 0:
            sorted_paras[prev_idx].nearby_raw_equation_ids.append(raw_eq_id)
        if next_idx < len(sorted_paras):
            sorted_paras[next_idx].nearby_raw_equation_ids.append(raw_eq_id)

    # --- cross-references ---
    cross_references = _extract_cross_references(paper_id, sections, paragraphs, sentences)

    return {
        "paper_id": paper_id,
        "title": title,
        "html_source": html_source,
        "schema_version": SCHEMA_VERSION,
        "sections": [s.to_dict() for s in sections],
        "paragraphs": {pid: p.to_dict() for pid, p in paragraphs.items()},
        "sentences": {sid: s.to_dict() for sid, s in sentences.items()},
        "raw_equations": [eq.to_dict() for eq in raw_equations],
        "cross_references": [r.to_dict() for r in cross_references],
    }


def _validate_document(doc: dict) -> None:
    paper_id = doc["paper_id"]
    para_ids = set(doc["paragraphs"])
    sent_ids = set(doc["sentences"])

    for para in doc["paragraphs"].values():
        for sid in para["sentence_ids"]:
            if sid not in sent_ids:
                raise ValueError(f"{paper_id}: sentence {sid!r} missing")

    seen_orders: set[int] = set()
    for para in doc["paragraphs"].values():
        order = para["document_order"]
        if order in seen_orders:
            raise ValueError(f"{paper_id}: duplicate document_order {order}")
        seen_orders.add(order)

    section_ids = {s["section_id"] for s in doc["sections"]}
    for para in doc["paragraphs"].values():
        if para["section_id"] not in section_ids:
            raise ValueError(f"{paper_id}: para {para['paragraph_id']} refs unknown section")


def _write_document(doc: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{doc['paper_id']}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)


def run() -> dict:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equations = json.loads(EQUATIONS_FILE.read_text(encoding="utf-8"))
    paper_ids = [pid for pid, entries in equations.items() if entries]

    missing = [pid for pid in paper_ids if not (HTML_DIR / f"{pid}.html").exists()]
    if missing:
        raise FileNotFoundError(f"Missing HTML for: {', '.join(missing)}")

    segmenter = _Segmenter()
    results = []

    for paper_id in paper_ids:
        html_file = HTML_DIR / f"{paper_id}.html"
        html_text = html_file.read_text(encoding="utf-8", errors="ignore")
        html_source = str(html_file.relative_to(HTML_DIR.parent.parent))
        doc = _parse_html(paper_id, html_text, html_source, segmenter)
        _validate_document(doc)
        _write_document(doc, DOCUMENTS_DIR)
        results.append({
            "paper_id": paper_id,
            "section_count": len(doc["sections"]),
            "paragraph_count": len(doc["paragraphs"]),
            "sentence_count": len(doc["sentences"]),
            "raw_equation_count": len(doc["raw_equations"]),
            "cross_reference_count": len(doc["cross_references"]),
        })

    report = {
        "paper_count": len(results),
        "total_raw_equations": sum(r["raw_equation_count"] for r in results),
        "total_sentences": sum(r["sentence_count"] for r in results),
        "papers": results,
    }
    return report
