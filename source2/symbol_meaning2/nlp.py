def load_spacy_pipeline():
	try:
		import spacy
		return spacy.load("en_core_web_sm")
	except (ImportError, OSError):
		return None
