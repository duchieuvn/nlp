from dataclasses import dataclass

from latex_utils import (
	DECORATORS,
	DROP_GROUP_COMMANDS,
	FORMATTING_COMMANDS,
	GREEK_COMMANDS,
	LIKELY_INDICES,
	NON_SYMBOL_COMMANDS,
	read_command,
	read_group,
	skip_space,
)
from symbol_models import SymbolCandidate


@dataclass(frozen=True)
class _Occurrence:
	canonical: str
	latex: str
	aliases: tuple[str, ...]


def _unique(values) -> list[str]:
	return list(dict.fromkeys(value for value in values if value))


def _subscript_name(content: str) -> str:
	content = content.strip()
	for command in (*FORMATTING_COMMANDS, *DROP_GROUP_COMMANDS):
		content = content.replace(f"\\{command}", "")
	parts = []
	position = 0
	while position < len(content):
		character = content[position]
		if character == "\\":
			command, end = read_command(content, position)
			if command in GREEK_COMMANDS:
				parts.append(command)
			position = max(end, position + 1)
		elif character.isalnum():
			end = position + 1
			while end < len(content) and content[end].isalnum():
				end += 1
			parts.append(content[position:end])
			position = end
		else:
			position += 1
	return "_".join(parts)


def _read_modifiers(text: str, position: int) -> tuple[str, str, bool, int]:
	subscript = ""
	superscript = ""
	prime = False
	while True:
		position = skip_space(text, position)
		if position >= len(text) or text[position] not in "_^":
			break
		kind = text[position]
		position = skip_space(text, position + 1)
		group = read_group(text, position)
		if group:
			content, position = group
		else:
			content = text[position:position + 1]
			position += bool(content)
		if kind == "_":
			subscript = _subscript_name(content)
		elif "prime" in content or content.strip() in {"'", "′"}:
			prime = True
		elif not content.strip().isdigit() and "dagger" not in content:
			superscript = _subscript_name(content)
	return subscript, superscript, prime, position


def _make_occurrence(
	base: str,
	latex_form: str,
	subscript: str = "",
	superscript: str = "",
	prime: bool = False,
	decorator: str = "",
) -> _Occurrence | None:
	canonical = base
	if subscript:
		canonical += f"_{subscript}"
	if superscript:
		canonical += f"_sup_{superscript}"
	if prime:
		canonical += "_prime"
	if decorator:
		canonical += f"_{decorator}"
	if base in LIKELY_INDICES:
		return None

	aliases = [canonical, latex_form]
	if base in GREEK_COMMANDS:
		unicode_base = GREEK_COMMANDS[base]
		plain_suffix = f"_{subscript}" if subscript else ""
		if superscript:
			plain_suffix += f"^{superscript}"
		prime_suffix = "'" if prime else ""
		aliases.extend([
			f"\\{base}{plain_suffix}{prime_suffix}",
			f"{base}{plain_suffix}{prime_suffix}",
			f"{unicode_base}{plain_suffix}{prime_suffix}",
		])
	elif prime:
		aliases.append(f"{base}'")
	return _Occurrence(canonical, latex_form, tuple(_unique(aliases)))


def _scan(text: str) -> list[_Occurrence]:
	occurrences = []
	position = 0
	while position < len(text):
		start = position
		character = text[position]
		if character == "\\":
			command, end = read_command(text, position)
			if not command:
				position += 1
				continue
			group_position = skip_space(text, end)
			group = read_group(text, group_position)
			if command in DROP_GROUP_COMMANDS and group:
				position = group[1]
				continue
			if command in FORMATTING_COMMANDS and group:
				occurrences.extend(_scan(group[0]))
				position = group[1]
				continue
			if command in DECORATORS and group:
				inner = _scan(group[0])
				position = group[1]
				subscript, superscript, prime, position = _read_modifiers(text, position)
				for item in inner:
					occurrence = _make_occurrence(
						item.canonical,
						text[start:position],
						subscript=subscript,
						superscript=superscript,
						prime=prime,
						decorator=DECORATORS[command],
					)
					if occurrence:
						occurrences.append(occurrence)
				continue
			if command in NON_SYMBOL_COMMANDS or command not in GREEK_COMMANDS:
				position = end
				continue
			base = command
			position = end
		elif character.isalpha():
			base = character
			position += 1
		else:
			position += 1
			continue

		subscript, superscript, prime, position = _read_modifiers(text, position)
		occurrence = _make_occurrence(
			base,
			text[start:position],
			subscript=subscript,
			superscript=superscript,
			prime=prime,
		)
		if occurrence:
			occurrences.append(occurrence)
	return occurrences


def extract_symbols(latex: str) -> list[SymbolCandidate]:
	grouped: dict[str, dict[str, list[str]]] = {}
	for occurrence in _scan(latex):
		entry = grouped.setdefault(
			occurrence.canonical,
			{"latex_forms": [], "aliases": []},
		)
		entry["latex_forms"].append(occurrence.latex)
		entry["aliases"].extend(occurrence.aliases)
	return [
		SymbolCandidate(
			canonical=canonical,
			latex_forms=_unique(values["latex_forms"]),
			aliases=_unique(values["aliases"]),
		)
		for canonical, values in grouped.items()
	]
