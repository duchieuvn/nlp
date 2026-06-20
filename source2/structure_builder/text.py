from copy import copy
import re


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
	return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_latex(latex: str) -> str:
	latex = latex.replace("%\n", "").replace("\n", " ")
	latex = re.sub(r"^\s*\\displaystyle\s*", "", latex)
	latex = re.sub(r"\\begin\{(?:aligned|split)\}", "", latex)
	latex = re.sub(r"\\end\{(?:aligned|split)\}", "", latex)
	latex = WHITESPACE_PATTERN.sub("", latex)
	return latex.rstrip(".,")


def clean_text_node(node) -> str:
	cleaned = copy(node)
	for math in cleaned.find_all("math"):
		alttext = math.get("alttext")
		math.replace_with(f" {alttext} " if alttext else " ")
	for noisy in cleaned.find_all(
		["annotation", "annotation-xml", "semantics", "script", "style"]
	):
		noisy.decompose()
	return normalize_text(cleaned.get_text(" ", strip=True))
