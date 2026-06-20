from dataclasses import dataclass

from models import SentenceRecord


@dataclass(frozen=True)
class SentenceSpan:
	text: str
	start: int
	end: int


class SpacySentenceSegmenter:
	def __init__(self) -> None:
		try:
			import spacy
		except ImportError as error:
			raise RuntimeError(
				"spaCy is required to build structured paper sentences."
			) from error
		self._nlp = spacy.blank("en")
		self._nlp.add_pipe("sentencizer")

	def spans(self, text: str) -> list[SentenceSpan]:
		if not text:
			return []
		doc = self._nlp(text)
		return [
			SentenceSpan(sentence.text, sentence.start_char, sentence.end_char)
			for sentence in doc.sents
			if sentence.text.strip()
		]

	def records(self, paragraph_id: str, text: str) -> list[SentenceRecord]:
		return [
			SentenceRecord(
				sentence_id=f"{paragraph_id}.s{index}",
				order=index,
				text=span.text,
				start=span.start,
				end=span.end,
			)
			for index, span in enumerate(self.spans(text), start=1)
		]
