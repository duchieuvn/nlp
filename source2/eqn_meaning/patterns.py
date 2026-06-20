import re


DISCOURSE_PREFIX = re.compile(
	r"^(?:therefore|thus|hence|consequently|in this case|so),?\s+",
	re.IGNORECASE,
)

ANCHOR_SUBJECT = re.compile(
	r"^(?P<subject>(?:the|a|an)\s+.{3,220}?)\s+"
	r"(?:can\s+be|may\s+be|is|are|was|were)\s+"
	r"(?:readily\s+)?(?:written|given|expressed|defined|represented|described|"
	r"evaluated|obtained)(?:\s+\w+){0,2}\s*(?:as|by|using|from)?\s*:?[.!]?$",
	re.IGNORECASE,
)

PLAIN_ADJACENT_SUBJECT = re.compile(
	r"^(?P<subject>(?:the|a|an)\s+.{3,180}?)\s+(?:is|are)\s*:?[.!]?$",
	re.IGNORECASE,
)

WE_OBJECT_ANCHOR = re.compile(
	r"^.*?\b(?:we|one)\s+(?:write\s+down|write|express|represent|formulate)\s+"
	r"(?P<object>(?:the|a|an)\s+.{3,180}?)\s+"
	r"(?:in\s+the\s+form|as|by)\s*:?[.!]?$",
	re.IGNORECASE,
)

DERIVED_OBJECT = re.compile(
	r"\b(?:derive|obtain|give|gives|yield|yields)\s+"
	r"(?P<object>(?:the|a|an)\s+.{3,160}?)[.!]?$",
	re.IGNORECASE,
)

EXPLICIT_DESCRIPTION_TEMPLATE = (
	r"\b(?:Eq(?:uation)?\.?\s*\(?\s*{equation_id}\s*\)?|"
	r"Equation\s*\(?\s*{equation_id}\s*\)?)\s+"
	r"(?:defines|describes|represents|gives|is|corresponds\s+to)\s+"
	r"(?P<object>.{{3,180}}?)[.!]?$"
)

DEFINITION_CUE = re.compile(
	r"\b(?:defines?|describes?|represents?|gives?|given\s+by|written\s+as|"
	r"expressed\s+as|known\s+as|called|corresponds\s+to|derive[ds]?|"
	r"obtain(?:ed|s)?|yield(?:s|ed)?)\b",
	re.IGNORECASE,
)

NAMED_CONCEPT = re.compile(
	r"\b(?:Hamiltonian|wave\s+function|characteristic\s+function|"
	r"covariance\s+matrix|density\s+matrix|dispersion\s+relation|"
	r"equation\s+of\s+motion|motion\s+equation|probability\s+distribution|"
	r"plane\s+wave|state\s+vector|transformation|displacement|"
	r"potential|energy|entropy|current|solution)\b",
	re.IGNORECASE,
)

PROCEDURAL_FRAGMENT = re.compile(
	r"\b(?:substitut|insert)\w*\b.*\b(?:obtain|give|yield)\s*:?[.!]?$",
	re.IGNORECASE,
)

INCOMPLETE_ENDING = re.compile(
	r"\b(?:as|by|from|form|is|are|obtain|obtained|give|given|yield|yields)"
	r"\s*:?[.!]?$",
	re.IGNORECASE,
)


def extract_span(
	text: str,
	equation_id: str,
	position: str,
) -> tuple[str, str]:
	cleaned = DISCOURSE_PREFIX.sub("", text.strip())
	anchor = ANCHOR_SUBJECT.match(cleaned)
	if anchor and position == "before_equation":
		return anchor.group("subject").strip(), "anchor_subject"
	plain_anchor = PLAIN_ADJACENT_SUBJECT.match(cleaned)
	if plain_anchor and position == "before_equation":
		return plain_anchor.group("subject").strip(), "anchor_subject"
	we_anchor = WE_OBJECT_ANCHOR.match(cleaned)
	if we_anchor and position == "before_equation":
		return we_anchor.group("object").strip(), "anchor_object"
	explicit = re.search(
		EXPLICIT_DESCRIPTION_TEMPLATE.format(equation_id=re.escape(equation_id)),
		cleaned,
		re.IGNORECASE,
	)
	if explicit:
		return explicit.group("object").strip(), "explicit_description"
	derived = DERIVED_OBJECT.search(cleaned)
	if derived:
		return derived.group("object").strip(), "derived_object"
	return cleaned, "source_sentence"
