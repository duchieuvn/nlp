"""Stage 7: Extract mathematical symbols from MathML, falling back to LaTeX."""

import json
import re

from bs4 import BeautifulSoup

from config import DOCUMENTS_DIR, EQUATIONS_DIR, OUTPUT_DIR, SYMBOLS_DIR

_GREEK_UNICODE = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}
_UNICODE_GREEK = {v: k for k, v in _GREEK_UNICODE.items()}
_OPERATORS = frozenset("""
    sum prod int oint iint iiint partial nabla times cdot otimes oplus leq geq
    neq approx equiv sim simeq cong propto infty to rightarrow leftarrow forall
    exists in notin subset superset subseteq supseteq ldots cdots Re Im det dim
    exp ln log lim max min sup inf arg ker tr rank diag span sin cos tan sinh cosh
    tanh text mathrm mathit operatorname begin end left right frac sqrt label tag
    nonumber overline underline
""".split())
_STRUCTURAL = _OPERATORS | frozenset(["displaystyle", "limits", "nolimits", "quad", "qquad"])
_PURE_INDICES = frozenset(["i", "j", "k", "l", "m", "n"])
_DECORATORS = {
    "^": "hat", "ˆ": "hat", "˙": "dot", "¨": "ddot", "¯": "bar",
    "→": "vec", "\u20d7": "vec", "~": "tilde", "˜": "tilde",
}
_LATEX_DECORATORS = frozenset([
    "hat", "bar", "tilde", "vec", "dot", "ddot", "breve", "check",
    "acute", "grave", "widehat", "widetilde",
])


def _latex_escape_symbol(base: str) -> str:
    if base in _GREEK_UNICODE:
        return "\\" + base
    return base


def _base_name(text: str) -> str:
    text = text.strip()
    if text in _UNICODE_GREEK:
        return _UNICODE_GREEK[text]
    if text.startswith("\\"):
        return text[1:]
    return text


def _clean_modifier(text: str) -> str:
    text = re.sub(r"\s+", "", text.strip())
    if text in _UNICODE_GREEK:
        return _UNICODE_GREEK[text]
    return text


def _upsert(symbols: dict[str, dict], base: str, modifiers: list[str], latex_forms: list[str]) -> None:
    """Insert or merge one canonical symbol entry.

    Parameters
    ----------
    symbols
        Mutable mapping from canonical symbol name to symbol metadata.
    base
        Base symbol name after Greek and LaTeX normalization.
    modifiers
        Subscript, superscript, or decoration descriptors.
    latex_forms
        LaTeX surface forms observed for the symbol.

    Returns
    -------
    None
        The ``symbols`` mapping is updated in place.
    """
    if not base or base in _OPERATORS or base.isdigit():
        return
    canonical_parts = [base]
    for modifier in modifiers:
        if ":" in modifier:
            canonical_parts.append(modifier.split(":", 1)[1])
    canonical = "_".join(p for p in canonical_parts if p)
    if not canonical:
        return
    entry = symbols.setdefault(canonical, {
        "canonical": canonical,
        "latex_forms": set(),
        "unicode": _GREEK_UNICODE.get(base, ""),
        "base": base,
        "modifiers": modifiers,
    })
    entry["latex_forms"].update(f for f in latex_forms if f)


def _node_text(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", "", node.get_text("", strip=True))


def _symbol_from_mathml_node(node) -> tuple[str, list[str], list[str]] | None:
    """Extract a symbol tuple from a supported MathML node.

    Parameters
    ----------
    node
        BeautifulSoup node such as ``mi``, ``msub``, ``msup``,
        ``msubsup``, ``mover``, or ``munder``.

    Returns
    -------
    tuple[str, list[str], list[str]] | None
        Base symbol, modifiers, and LaTeX forms when the node represents
        a symbol; otherwise ``None``.
    """
    if node is None or not getattr(node, "name", None):
        return None
    name = node.name.lower()

    if name == "mi":
        base = _base_name(_node_text(node))
        if not base or base in _OPERATORS:
            return None
        return base, [], [_latex_escape_symbol(base)]

    children = [c for c in node.find_all(recursive=False) if getattr(c, "name", None)]
    if name in {"mstyle", "mrow", "semantics", "math"} and len(children) == 1:
        return _symbol_from_mathml_node(children[0])

    if name in {"msub", "msup", "msubsup"} and len(children) >= 2:
        base_info = _symbol_from_mathml_node(children[0])
        if not base_info:
            return None
        base, modifiers, forms = base_info
        sub = _clean_modifier(_node_text(children[1])) if name in {"msub", "msubsup"} else ""
        sup_idx = 2 if name == "msubsup" else 1
        sup = _clean_modifier(_node_text(children[sup_idx])) if name in {"msup", "msubsup"} and len(children) > sup_idx else ""
        new_mods = list(modifiers)
        latex_forms = list(forms)
        if sub:
            new_mods.append(f"subscript:{sub}")
            latex_forms = [f"{f}_{{{sub}}}" for f in latex_forms]
            if len(sub) == 1:
                latex_forms.extend(f"{f}_{sub}" for f in forms)
        if sup and not (sup.isdigit() and not sub):
            new_mods.append(f"superscript:{sup}")
            latex_forms = [f"{f}^{{{sup}}}" for f in latex_forms]
        elif sup:
            latex_forms = [f"{f}^{{{sup}}}" for f in latex_forms]
        return base, new_mods, latex_forms

    if name in {"mover", "munder"} and len(children) >= 2:
        base_info = _symbol_from_mathml_node(children[0])
        if not base_info:
            return None
        base, modifiers, forms = base_info
        deco = _DECORATORS.get(_node_text(children[1]), _node_text(children[1]) or name)
        return base, modifiers + [f"decorator:{deco}"], [f"\\{deco}{{{f}}}" for f in forms]

    return None


def _extract_symbols_from_mathml(mathml_forms: list[str]) -> list[dict]:
    symbols: dict[str, dict] = {}
    for markup in mathml_forms:
        soup = BeautifulSoup(markup, "html.parser")
        for node in soup.find_all(["mi", "msub", "msup", "msubsup", "mover", "munder"]):
            # Composite parents own their child mi; skip child nodes inside a supported composite.
            parent = node.find_parent(["msub", "msup", "msubsup", "mover", "munder"])
            if parent is not None and parent is not node:
                continue
            info = _symbol_from_mathml_node(node)
            if info:
                _upsert(symbols, *info)
    return _finalize(symbols)


def _parse_braced_arg(latex: str, pos: int) -> tuple[str, int]:
    """Read one LaTeX argument starting at a position.

    Parameters
    ----------
    latex
        LaTeX string being scanned.
    pos
        Current index, usually at an opening brace or command.

    Returns
    -------
    tuple[str, int]
        Parsed argument text and the next position after the argument.
    """
    if pos >= len(latex):
        return "", pos
    if latex[pos] != "{":
        if latex[pos] == "\\":
            m = re.match(r"\\[a-zA-Z]+", latex[pos:])
            if m:
                return m.group(0), pos + m.end()
        return latex[pos], pos + 1
    depth = 0
    for i in range(pos, len(latex)):
        if latex[i] == "{":
            depth += 1
        elif latex[i] == "}":
            depth -= 1
            if depth == 0:
                return latex[pos + 1:i], i + 1
    return latex[pos + 1:], len(latex)


def _skip_spaces(latex: str, pos: int) -> int:
    while pos < len(latex) and latex[pos].isspace():
        pos += 1
    return pos


def _read_sub_sup(latex: str, pos: int) -> tuple[int, str, str]:
    sub = ""
    sup = ""
    while pos < len(latex):
        pos = _skip_spaces(latex, pos)
        if pos < len(latex) and latex[pos] == "_" and not sub:
            sub, pos = _parse_braced_arg(latex, _skip_spaces(latex, pos + 1))
        elif pos < len(latex) and latex[pos] == "^" and not sup:
            sup, pos = _parse_braced_arg(latex, _skip_spaces(latex, pos + 1))
        else:
            break
    return pos, sub.strip(), sup.strip()


def _extract_symbols_from_latex(latex: str) -> list[dict]:
    """Extract symbols from LaTeX when MathML is unavailable.

    Parameters
    ----------
    latex
        Equation LaTeX string from the selected equation record.

    Returns
    -------
    list[dict]
        Finalized canonical symbol records, including base symbols,
        modifiers, Unicode Greek forms, and observed LaTeX forms.
    """
    latex = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", latex)
    latex = re.sub(r"\\(?:text|mathrm|mathit|operatorname)\{[^}]*\}", " ", latex)
    symbols: dict[str, dict] = {}
    pos = 0
    while pos < len(latex):
        pos = _skip_spaces(latex, pos)
        if pos >= len(latex):
            break
        c = latex[pos]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", latex[pos:])
            if not m:
                pos += 1
                continue
            cmd = m.group(1)
            end = pos + m.end()
            if cmd in _LATEX_DECORATORS:
                arg, arg_end = _parse_braced_arg(latex, _skip_spaces(latex, end))
                inner = _extract_symbols_from_latex(arg)
                for s in inner:
                    decorated_forms = [f"\\{cmd}{{{form}}}" for form in s["latex_forms"]]
                    _upsert(symbols, s["base"], [f"decorator:{cmd}"] + s["modifiers"], decorated_forms)
                pos = arg_end
                continue
            if cmd in _STRUCTURAL:
                pos = end
                continue
            pos, sub, sup = _read_sub_sup(latex, end)
            _add_latex_token(symbols, cmd, sub, sup, "\\" + cmd)
        elif c.isalpha():
            pos, sub, sup = _read_sub_sup(latex, pos + 1)
            _add_latex_token(symbols, c, sub, sup, c)
        else:
            pos += 1
    return _finalize(symbols)


def _add_latex_token(symbols: dict[str, dict], base: str, sub: str, sup: str, form_base: str) -> None:
    """Add one parsed LaTeX token to the symbol mapping.

    Parameters
    ----------
    symbols
        Mutable canonical symbol mapping.
    base
        Base token name without subscript or superscript.
    sub
        Parsed subscript text, if present.
    sup
        Parsed superscript text, if present.
    form_base
        Original LaTeX form for the base token.

    Returns
    -------
    None
        The ``symbols`` mapping is updated in place.
    """
    if base in _OPERATORS or base in _STRUCTURAL:
        return
    modifiers = []
    forms = [form_base]
    if sub:
        sub_clean = _clean_modifier(re.sub(r"\\[A-Za-z]+", "", sub) or sub)
        modifiers.append(f"subscript:{sub_clean}")
        forms = [f"{f}_{{{sub}}}" for f in forms]
        if len(sub) == 1:
            forms.append(f"{form_base}_{sub}")
    if sup:
        sup_clean = _clean_modifier(re.sub(r"\\[A-Za-z]+", "", sup) or sup)
        if not (sup_clean.isdigit() and not sub):
            modifiers.append(f"superscript:{sup_clean}")
        forms = [f"{f}^{{{sup}}}" for f in forms]
    _upsert(symbols, base, modifiers, forms)


def _finalize(symbols: dict[str, dict]) -> list[dict]:
    result = []
    for symbol in symbols.values():
        if not symbol["modifiers"] and len(symbol["base"]) == 1 and symbol["base"] in _PURE_INDICES:
            continue
        symbol["latex_forms"] = sorted(symbol["latex_forms"])
        result.append(symbol)
    return sorted(result, key=lambda s: s["canonical"])


def _process_paper(paper_id: str) -> dict:
    """Extract symbols for every selected equation in one paper.

    Parameters
    ----------
    paper_id
        arXiv identifier whose document and equation artifacts should be
        processed.

    Returns
    -------
    dict
        Stage 7 payload with symbol lists aligned to equation IDs.
    """
    eq_data = json.loads((EQUATIONS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    doc_path = DOCUMENTS_DIR / f"{paper_id}.json"
    raw_by_id = {}
    if doc_path.exists():
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        raw_by_id = {e["raw_equation_id"]: e for e in doc.get("raw_equations", [])}

    results = []
    for eq in eq_data.get("equations", []):
        equation_id = eq["equation_id"]
        if eq.get("match_method") == "unresolved":
            results.append({"equation_id": equation_id, "symbols": []})
            continue
        raw = raw_by_id.get(eq.get("raw_equation_id"), {})
        symbols = _extract_symbols_from_mathml(raw.get("mathml", []))
        if not symbols:
            symbols = _extract_symbols_from_latex(eq.get("latex", ""))
        results.append({"equation_id": equation_id, "symbols": symbols})
    return {"paper_id": paper_id, "equations": results}


def run() -> dict:
    """Run Stage 7 over all selected-equation files.

    Returns
    -------
    dict
        Summary counts for processed papers and extracted symbols.

    Raises
    ------
    FileNotFoundError
        If no Stage 4 equation files are available.
    """
    SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eq_files = sorted(EQUATIONS_DIR.glob("*.json"))
    if not eq_files:
        raise FileNotFoundError(f"No equation files in {EQUATIONS_DIR}")

    total_symbols = 0
    paper_results = []
    for eq_file in eq_files:
        paper_id = eq_file.stem
        result = _process_paper(paper_id)
        out = SYMBOLS_DIR / f"{paper_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)

        count = sum(len(e["symbols"]) for e in result["equations"])
        total_symbols += count
        paper_results.append({"paper_id": paper_id, "symbol_count": count})

    return {
        "paper_count": len(eq_files),
        "total_symbols": total_symbols,
        "papers": paper_results,
    }
