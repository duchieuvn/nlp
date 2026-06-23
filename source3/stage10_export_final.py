"""Stage 10: Join all pipeline artifacts and write data/final_data.json."""

import json

from config import (
    EQUATIONS_DIR,
    FINAL_DATA_FILE,
    MEANINGS_DIR,
    OUTPUT_DIR,
    PAPER_LIST_FILE,
    RELATIONS_DIR,
    SYMBOL_MEANINGS_DIR,
    SYMBOLS_DIR,
)

_VALID_GRADES = {"strong", "potential", "none"}
_VALID_DESCRIPTIONS = {
    "explicit citation", "derived from", "equivalent", "special case",
    "shares symbols", "same section context", "none",
}


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _load_paper_artifact(directory, paper_id: str) -> dict | None:
    path = directory / f"{paper_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _index_equations(artifact: dict | None) -> dict[str, dict]:
    if not artifact:
        return {}
    return {
        eq["equation_id"]: eq
        for eq in artifact.get("equations", [])
        if isinstance(eq, dict) and eq.get("equation_id")
    }


def _load_paper_equations(paper_id: str) -> dict[str, dict]:
    """Return { equation_id: {"equation": latex} } from Stage 4 output."""
    artifact = _load_paper_artifact(EQUATIONS_DIR, paper_id)
    if not artifact:
        return {}
    return {
        eq["equation_id"]: {"equation": eq.get("latex", "")}
        for eq in artifact.get("equations", [])
        if isinstance(eq, dict) and eq.get("equation_id")
    }


def _latex_symbol_key(symbol: dict) -> str:
    """Return the LaTeX display key for a symbol, falling back to canonical."""
    latex_forms = symbol.get("latex_forms") or []
    if latex_forms:
        return latex_forms[0]
    return symbol["canonical"]


def _build_symbols_per_equation(
    sym_data: dict | None,
    sym_meaning_data: dict | None,
) -> dict[str, dict[str, str]]:
    """Return { equation_id: { latex_symbol: definition } } for each equation's own symbols."""
    if not sym_data:
        return {}

    # Build meaning lookup by equation_id so definitions do not leak across equations.
    sym_meaning_index: dict[str, dict[str, str]] = {}
    if sym_meaning_data:
        for eq in sym_meaning_data["equations"]:
            eq_id = eq["equation_id"]
            sym_meaning_index.setdefault(eq_id, {})
            for canonical, info in eq.get("symbol_meanings", {}).items():
                sym_meaning_index[eq_id][canonical] = info["definition"]

    result: dict[str, dict[str, str]] = {}
    for eq in sym_data["equations"]:
        equation_id = eq["equation_id"]
        eq_syms: dict[str, str] = {}
        definitions = sym_meaning_index.get(equation_id, {})
        for sym in eq.get("symbols", []):
            canonical = sym["canonical"]
            if canonical in definitions:
                eq_syms[_latex_symbol_key(sym)] = definitions[canonical]
        result[equation_id] = eq_syms
    return result


def _summarize_symbols(symbols: list[dict]) -> str:
    names = [_latex_symbol_key(sym) for sym in symbols]
    if not names:
        return "Detected no symbols in the equation."
    preview = ", ".join(names[:6])
    if len(names) > 6:
        preview += f", and {_plural(len(names) - 6, 'other')}"
    return f"Detected {preview} in the equation."


def _build_audit_trail(
    eq_id: str,
    aligned_eq: dict | None,
    meaning_eq: dict | None,
    symbol_eq: dict | None,
    symbol_meaning_eq: dict | None,
    eq_relations: dict,
    reviewed_count: int,
) -> dict[str, str]:
    label = f"({eq_id})"
    if aligned_eq:
        method = aligned_eq.get("match_method", "unknown")
        if method == "unresolved":
            extract_eq = (
                f"Found reviewed equation {label}, but stage04 could not align it "
                "to a raw HTML equation."
            )
        else:
            extract_eq = f"Found numbered equation {label} in the source text using {method}."
    else:
        extract_eq = f"Found reviewed equation {label} in the annotation data."

    before_count = len((aligned_eq or {}).get("before_sentences", []))
    after_count = len((aligned_eq or {}).get("after_sentences", []))
    context_count = before_count + after_count
    if context_count:
        extract_context = (
            f"Collected {_plural(before_count, 'preceding sentence')} and "
            f"{_plural(after_count, 'following sentence')} as local context."
        )
    elif aligned_eq and aligned_eq.get("match_method") == "unresolved":
        extract_context = "No local context was attached because the equation alignment was unresolved."
    else:
        extract_context = "No nearby context sentences were available for this equation."

    raw_symbols = (symbol_eq or {}).get("symbols", [])
    symbol_meanings = (symbol_meaning_eq or {}).get("symbol_meanings", {})
    relation_values = list(eq_relations.values())
    nonempty_relations = [r for r in relation_values if r.get("grade") != "none"]

    meaning = (meaning_eq or {}).get("meaning", "")
    match_source = (meaning_eq or {}).get("match_source", "none")
    score = (meaning_eq or {}).get("score", 0.0)
    if meaning:
        meaning_summary = f"Selected equation meaning from {match_source} evidence with score {score}."
    else:
        meaning_summary = "No equation-level meaning was selected from the available evidence."

    if symbol_meanings:
        extract_symbol_name = (
            f"Assigned definitions to {_plural(len(symbol_meanings), 'symbol')} "
            f"after retrieval-based symbol lookup. {meaning_summary}"
        )
    elif raw_symbols:
        extract_symbol_name = (
            "No symbol definitions were assigned after retrieval-based symbol lookup. "
            f"{meaning_summary}"
        )
    else:
        extract_symbol_name = f"No symbol definitions were needed because no symbols were detected. {meaning_summary}"

    expected_relations = max(reviewed_count - 1, 0)
    if expected_relations:
        build_relations = (
            f"Built {_plural(len(eq_relations), 'relation')} covering the other "
            f"{_plural(expected_relations, 'reviewed equation')}; "
            f"{_plural(len(nonempty_relations), 'non-none relation')} remained after scoring."
        )
    else:
        build_relations = "No relation pairs were needed because this paper has one reviewed equation."

    return {
        "extract_eq": extract_eq,
        "extract_context": extract_context,
        "find_symbol": _summarize_symbols(raw_symbols),
        "extract_symbol_name": extract_symbol_name,
        "build_relations": build_relations,
        "validate_entry": (
            "Confirmed required fields equation, meaning, symbols, relations, "
            "and audit-trail are present."
        ),
    }


def _validate_paper(paper_id: str, equations: dict, paper_obj: dict, sym_data: dict | None) -> list[str]:
    """Return list of validation errors."""
    errors = []
    reviewed_ids = set(equations.keys())
    if set(paper_obj.keys()) != reviewed_ids:
        errors.append(f"{paper_id}: final object does not cover all reviewed equations")

    symbol_keys: dict[str, set[str]] = {}
    if sym_data:
        for eq in sym_data.get("equations", []):
            canonicals = [s["canonical"] for s in eq.get("symbols", [])]
            if len(canonicals) != len(set(canonicals)):
                errors.append(f"{paper_id}/{eq['equation_id']}: duplicate symbol canonicals")
            latex_keys = [_latex_symbol_key(s) for s in eq.get("symbols", [])]
            if len(latex_keys) != len(set(latex_keys)):
                errors.append(f"{paper_id}/{eq['equation_id']}: duplicate symbol LaTeX keys")
            symbol_keys[eq["equation_id"]] = set(latex_keys)

    for eq_id, eq_obj in paper_obj.items():
        if eq_obj == {}:
            continue
        if not isinstance(eq_obj.get("audit-trail"), dict):
            errors.append(f"{paper_id}/{eq_id}: missing audit-trail")
        # equation field must match reviewed latex
        reviewed_latex = equations.get(eq_id, {}).get("equation", "")
        if eq_obj.get("equation", "") != reviewed_latex:
            errors.append(f"{paper_id}/{eq_id}: equation text mismatch")

        unknown_symbols = set(eq_obj.get("symbols", {})) - symbol_keys.get(eq_id, set())
        if unknown_symbols:
            errors.append(f"{paper_id}/{eq_id}: symbols not extracted for equation {sorted(unknown_symbols)[:3]}")

        # relations must cover all other reviewed equations
        relation_keys = set(eq_obj.get("relations", {}).keys())
        expected = reviewed_ids - {eq_id}
        missing = expected - relation_keys
        if missing:
            errors.append(f"{paper_id}/{eq_id}: missing relations for {sorted(missing)[:3]}")

        for target_id, rel in eq_obj.get("relations", {}).items():
            if rel.get("grade") not in _VALID_GRADES:
                errors.append(f"{paper_id}/{eq_id}→{target_id}: invalid grade {rel.get('grade')!r}")
            if rel.get("description") not in _VALID_DESCRIPTIONS:
                errors.append(
                    f"{paper_id}/{eq_id}→{target_id}: invalid description {rel.get('description')!r}"
                )

    return errors


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Paper order derived from Stage 4 output (already ordered and capped at target)
    eq_report_path = OUTPUT_DIR / "equation_alignment_report.json"
    if eq_report_path.exists():
        eq_report = json.loads(eq_report_path.read_text(encoding="utf-8"))
        ordered_paper_ids = [r["paper_id"] for r in eq_report.get("papers", [])]
    else:
        lines = PAPER_LIST_FILE.read_text(encoding="utf-8").splitlines()
        ordered_paper_ids = [
            line.removeprefix("arXiv:").strip() for line in lines if line.strip()
        ]

    final_data: dict = {}
    validation_errors: list[str] = []
    stats = {"papers": 0, "equations": 0, "with_meaning": 0, "with_symbols": 0}

    for paper_id in ordered_paper_ids:
        reviewed = _load_paper_equations(paper_id)

        # Papers with no reviewed equations → empty object
        if not reviewed:
            final_data[paper_id] = {}
            continue

        # Load artifacts
        aligned = _load_paper_artifact(EQUATIONS_DIR, paper_id)
        meanings = _load_paper_artifact(MEANINGS_DIR, paper_id)
        symbols = _load_paper_artifact(SYMBOLS_DIR, paper_id)
        sym_meanings = _load_paper_artifact(SYMBOL_MEANINGS_DIR, paper_id)
        relations = _load_paper_artifact(RELATIONS_DIR, paper_id)

        # Papers that failed HTML (no meanings artifact) → empty object
        if meanings is None:
            final_data[paper_id] = {}
            continue

        # Build meaning index: equation_id -> meaning text
        meaning_index = {
            e["equation_id"]: e.get("meaning", "")
            for e in meanings.get("equations", [])
        }
        aligned_index = _index_equations(aligned)
        meaning_records = _index_equations(meanings)
        symbol_records = _index_equations(symbols)
        symbol_meaning_records = _index_equations(sym_meanings)

        # Build per-equation symbol definitions
        symbols_per_eq = _build_symbols_per_equation(symbols, sym_meanings)

        # Build relation index: equation_id -> { target_id: {grade, description} }
        relation_index: dict[str, dict] = (relations or {}).get("relations", {})

        paper_obj: dict = {}
        for eq_id, entry in reviewed.items():
            eq_relations = dict(relation_index.get(eq_id, {}))
            # Ensure every other reviewed equation is covered
            for other_id in reviewed:
                if other_id != eq_id and other_id not in eq_relations:
                    eq_relations[other_id] = {"grade": "none", "description": "none"}

            eq_symbols = symbols_per_eq.get(eq_id, {})
            paper_obj[eq_id] = {
                "equation": entry.get("equation", ""),
                "meaning": meaning_index.get(eq_id, ""),
                "symbols": eq_symbols,
                "relations": eq_relations,
                "audit-trail": _build_audit_trail(
                    eq_id=eq_id,
                    aligned_eq=aligned_index.get(eq_id),
                    meaning_eq=meaning_records.get(eq_id),
                    symbol_eq=symbol_records.get(eq_id),
                    symbol_meaning_eq=symbol_meaning_records.get(eq_id),
                    eq_relations=eq_relations,
                    reviewed_count=len(reviewed),
                ),
            }

            stats["equations"] += 1
            if meaning_index.get(eq_id):
                stats["with_meaning"] += 1
            if eq_symbols:
                stats["with_symbols"] += 1

        errors = _validate_paper(paper_id, reviewed, paper_obj, symbols)
        validation_errors.extend(errors)

        final_data[paper_id] = paper_obj
        stats["papers"] += 1

    if validation_errors:
        raise ValueError(
            f"Validation failed with {len(validation_errors)} error(s):\n"
            + "\n".join(validation_errors[:20])
        )

    tmp = FINAL_DATA_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(final_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(FINAL_DATA_FILE)

    return {
        "paper_count": stats["papers"],
        "equation_count": stats["equations"],
        "meanings_found": stats["with_meaning"],
        "output": str(FINAL_DATA_FILE),
    }
