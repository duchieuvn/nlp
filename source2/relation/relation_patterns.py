import re


DERIVATION_CUE = re.compile(
	r"\b(?:substitut\w*|insert\w*|using|use[ds]?|follows?\s+from|"
	r"deriv(?:e|ed|es|ing|ation)|obtain\w*|yield\w*|rewrit\w*|"
	r"integrat\w*|evaluat\w*)\b",
	re.IGNORECASE,
)
EQUIVALENCE_CUE = re.compile(
	r"\b(?:equivalent\s+to|same\s+as|identical\s+to|corresponds?\s+to)\b",
	re.IGNORECASE,
)
SPECIAL_CASE_CUE = re.compile(
	r"\b(?:special\s+case|reduces?\s+to|in\s+the\s+limit|"
	r"by\s+setting|setting\s+\S+\s*(?:=|to)|for\s+zero)\b",
	re.IGNORECASE,
)


def mentions_equation(text: str, equation_id: str) -> bool:
	return equation_reference_match(text, equation_id) is not None


def equation_reference_match(text: str, equation_id: str):
	identifier = re.escape(equation_id)
	return re.search(
		rf"\b(?:Eq(?:uation)?s?\.?|Equation)\s*"
		rf"(?:\(\s*)?{identifier}(?:\s*\))?(?![\d.])",
		text,
		re.IGNORECASE,
	)


def cue_near_equation(cue: re.Pattern, text: str, equation_id: str) -> bool:
	reference = equation_reference_match(text, equation_id)
	if reference is None:
		return False
	for match in cue.finditer(text):
		distance = max(
			0,
			reference.start() - match.end(),
			match.start() - reference.end(),
		)
		if distance <= 45:
			return True
	return False


def from_equation_cue(text: str, equation_id: str) -> bool:
	reference = equation_reference_match(text, equation_id)
	if reference is None:
		return False
	return bool(re.search(
		rf"\bfrom\s+(?:the\s+)?{re.escape(reference.group(0))}",
		text,
		re.IGNORECASE,
	))
