import re


SPACE = r"\s*"
DEFINITION_CUE = re.compile(
	r"\b(?:denotes?|represents?|means?|is\s+defined\s+as|are\s+defined\s+as|"
	r"is|are|refers?\s+to|stands?\s+for)\b",
	re.IGNORECASE,
)
VAGUE_DEFINITION = re.compile(
	r"^(?:it|this|that|these|those|aforementioned|the\s+(?:same|above|following))\b",
	re.IGNORECASE,
)
PROPERTY_DEFINITION = re.compile(
	r"^(?:applied|discretized|equal\s+to|fully\s+characterized|given\s+by|"
	r"held\s+fixed|implemented|irreducible|less\s+than|more\s+likely|much\s+less|"
	r"in\s+Eq\b|non-?zero\b|on\s+the|provided|proportional\s+to|real\b|"
	r"sufficiently\s+small)",
	re.IGNORECASE,
)


def _alias_pattern(aliases: list[str]) -> str:
	parts = []
	for alias in sorted(set(aliases), key=len, reverse=True):
		alias = alias.strip()
		if not alias:
			continue
		part = re.escape(alias)
		if alias[0].isalnum():
			part = rf"(?<![A-Za-z0-9]){part}"
		if alias[-1].isalnum():
			part = rf"{part}(?![A-Za-z0-9_^])"
		parts.append(part)
	return "(?:" + "|".join(parts) + ")"


def mentions_alias(text: str, aliases: list[str]) -> bool:
	pattern = _alias_pattern(aliases)
	return pattern != "(?:)" and bool(re.search(pattern, text))


def _clean_definition(value: str) -> str:
	value = re.sub(r"\s+", " ", value).strip(" \t,;:.()")
	value = re.split(
		r"(?:,\s+(?:and\s+)?|\s+and\s+)(?:\\?[A-Za-z][^,]{0,45}?)\s+"
		r"(?:is|are|denotes?|represents?)\b",
		value,
		maxsplit=1,
		flags=re.IGNORECASE,
	)[0]
	return re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE).strip()


def is_valid_definition(value: str, aliases: list[str]) -> bool:
	words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", value)
	if (
		not words
		or len(words) > 30
		or VAGUE_DEFINITION.search(value)
		or PROPERTY_DEFINITION.search(value)
		or re.search(r"\b(?:Here|Where),?\s+we\b", value)
	):
		return False
	if value.casefold() in {alias.strip().casefold() for alias in aliases}:
		return False
	math_characters = sum(character in r"\{}_^=" for character in value)
	balanced = all(
		value.count(opening) == value.count(closing)
		for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"))
	)
	return balanced and math_characters / max(1, len(value)) <= 0.35


def extract_definition(text: str, aliases: list[str]) -> tuple[str, str, str]:
	alias = _alias_pattern(aliases)
	if alias == "(?:)":
		return "", "no_pattern", ""
	patterns = (
		(
			"symbol_before_cue",
			rf"(?P<alias>{alias}){SPACE}(?:,?{SPACE})"
			r"(?:denotes?|represents?|means?|refers?\s+to|stands?\s+for|"
			r"is\s+defined\s+as|are\s+defined\s+as)\s+"
			r"(?P<definition>[^.;]{1,220})",
		),
		(
			"contextual_copula",
			r"\b(?:where|here|with)\b\s*[^.;]{0,180}?"
			rf"(?P<alias>{alias})\s+(?:is|are)\s+"
			r"(?P<definition>[^.;]{1,220})",
		),
		(
			"initial_copula",
			rf"^\s*(?P<alias>{alias})\s+(?:is|are)\s+"
			r"(?P<definition>(?:the|a|an)\s+[^.;]{{1,220}})",
		),
		(
			"definition_before_symbol",
			r"(?:we\s+)?(?:denote|represent|define|write)\s+"
			r"(?P<definition>[^.;]{1,220}?)\s+(?:as|by)\s+"
			rf"(?P<alias>{alias})",
		),
		(
			"passive_definition",
			r"(?P<definition>[^.;]{1,180}?)\s+(?:is|are)\s+"
			r"(?:denoted|represented|defined)\s+(?:by|as)\s+"
			rf"(?P<alias>{alias})",
		),
	)
	for strategy, pattern in patterns:
		match = re.search(pattern, text, re.IGNORECASE)
		if not match:
			continue
		definition_text = match.group("definition")
		if strategy == "definition_before_symbol" and " as " in definition_text:
			definition_text = re.split(
				r"\band\b", definition_text, flags=re.IGNORECASE
			)[-1]
		definition = _clean_definition(definition_text)
		if is_valid_definition(definition, aliases):
			return definition, strategy, match.group("alias")
	return "", "no_pattern", ""
