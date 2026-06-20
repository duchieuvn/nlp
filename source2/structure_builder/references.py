import re


REFERENCE_PATTERN = re.compile(
	r"\b(?P<prefix>Eqs?\.|Equations?)\s*"
	r"(?P<body>\([^)]{1,80}\)(?:\s*[-–—]\s*\([^)]{1,40}\))?|"
	r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)",
	re.IGNORECASE,
)
LABEL_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")
RANGE_PATTERN = re.compile(
	r"^\(?\s*(\d+)\s*\)?\s*[-–—]\s*\(?\s*(\d+)\s*\)?$"
)


def parse_reference_labels(body: str) -> tuple[str, list[str]]:
	range_match = RANGE_PATTERN.match(body.strip())
	if range_match:
		start, end = map(int, range_match.groups())
		if start <= end and end - start <= 100:
			return "range", [str(value) for value in range(start, end + 1)]
	labels = LABEL_PATTERN.findall(body)
	reference_type = "list" if len(labels) > 1 else "singular"
	return reference_type, labels


def iter_explicit_references(text: str):
	for match in REFERENCE_PATTERN.finditer(text):
		reference_type, labels = parse_reference_labels(match.group("body"))
		if labels:
			yield match, reference_type, labels
