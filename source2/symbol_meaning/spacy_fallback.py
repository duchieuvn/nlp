from patterns import is_valid_definition


def load_spacy_pipeline():
	try:
		import spacy
		return spacy.load("en_core_web_sm")
	except (ImportError, OSError):
		return None


def extract_dependency_definition(
	text: str,
	aliases: list[str],
	nlp,
) -> tuple[str, str]:
	if nlp is None:
		return "", ""
	plain_aliases = {
		alias.casefold()
		for alias in aliases
		if alias and all(character.isalnum() or character in "_-'" for character in alias)
	}
	if not plain_aliases:
		return "", ""
	document = nlp(text)
	for token in document:
		if token.text.casefold() not in plain_aliases:
			continue
		if token.dep_ == "nsubj" and token.head.pos_ in {"NOUN", "PROPN"}:
			if any(child.dep_ == "cop" for child in token.head.children):
				span = document[
					min(item.i for item in token.head.subtree):
					max(item.i for item in token.head.subtree) + 1
				].text
				span = span.replace(token.text, "", 1).strip(" ,.;:")
				if is_valid_definition(span, aliases):
					return span, token.text
		for child in token.children:
			if child.dep_ == "appos" and child.pos_ in {"NOUN", "PROPN"}:
				span = document[
					min(item.i for item in child.subtree):
					max(item.i for item in child.subtree) + 1
				].text.strip(" ,.;:")
				if is_valid_definition(span, aliases):
					return span, token.text
	return "", ""
