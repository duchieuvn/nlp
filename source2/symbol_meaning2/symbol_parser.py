import re

from .symbol_models import ParsedSymbol


DECORATORS = ("vec", "bar", "hat", "tilde", "dot", "ddot", "overline", "mathbf")


def canonicalize_latex(value: str) -> str:
	value = re.sub(r"\\(?:left|right)", "", value)
	value = re.sub(r"\s+", "", value)
	value = re.sub(r"_\{([^{}]+)\}", r"_\1", value)
	value = re.sub(r"\^\{([^{}]+)\}", r"^\1", value)
	return value


def _atom(value: str, start: int) -> tuple[str, int]:
	if start >= len(value):
		return "", start
	if value[start] != "{":
		if value[start] == "\\":
			match = re.match(r"\\[A-Za-z]+|\\.", value[start:])
			return (match.group(0), start + len(match.group(0))) if match else ("", start)
		return value[start], start + 1
	depth = 0
	for index in range(start, len(value)):
		depth += value[index] == "{"
		depth -= value[index] == "}"
		if depth == 0:
			return value[start:index + 1], index + 1
	return value[start:], len(value)


def _ungroup(value: str | None) -> str | None:
	if not value:
		return None
	value = value.strip()
	while len(value) > 1 and value.startswith("{") and value.endswith("}"):
		value = value[1:-1].strip()
	return value or None


def parse_symbol(symbol: dict, equation: str = "") -> ParsedSymbol:
	forms = list(dict.fromkeys(symbol.get("latex_forms", [])))
	original = forms[0] if forms else symbol.get("canonical", "")
	working = re.sub(r"\\(?:left|right)", "", original.strip())
	decorators = []
	while True:
		match = re.match(r"\\(" + "|".join(DECORATORS) + r")\s*", working)
		if not match:
			break
		decorators.append(match.group(1))
		value, end = _atom(working, match.end())
		working = (_ungroup(value) or "") + working[end:]
	start = min((position for position in (working.find("_"), working.find("^")) if position >= 0), default=len(working))
	base = _ungroup(working[:start]) or symbol.get("canonical", "")
	subscript = superscript = None
	position = start
	while position < len(working):
		marker = working[position]
		if marker not in "_^":
			position += 1
			continue
		value, position = _atom(working, position + 1)
		if marker == "_":
			subscript = _ungroup(value)
		else:
			superscript = _ungroup(value)
	return ParsedSymbol(
		original, base, subscript, superscript, decorators,
		symbol.get("canonical", ""), equation,
		list(dict.fromkeys(symbol.get("aliases", []))),
	)
