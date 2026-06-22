from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re

from .symbol_parser import canonicalize_latex


@dataclass(frozen=True)
class InlineMathOccurrence:
	latex: str
	canonical: str
	paragraph: str
	dom_id: str | None
	paragraph_id: str | None


class _Parser(HTMLParser):
	def __init__(self):
		super().__init__(convert_charrefs=True)
		self.parts = []
		self.pending = []
		self.records = []
		self.paragraph_id = None
		self.math_depth = 0

	def handle_starttag(self, tag, attrs):
		values = dict(attrs)
		if tag == "p":
			self._flush()
			self.paragraph_id = values.get("id")
		if tag == "math" and values.get("display", "inline") == "inline":
			latex = values.get("alttext", "").strip()
			if latex:
				self.parts.append(f" ${latex}$ ")
				self.pending.append((latex, values.get("id")))
			self.math_depth += 1

	def handle_endtag(self, tag):
		if tag == "math" and self.math_depth:
			self.math_depth -= 1
		if tag == "p":
			self._flush()

	def handle_data(self, data):
		if not self.math_depth:
			self.parts.append(data)

	def _flush(self):
		paragraph = re.sub(r"\s+", " ", "".join(self.parts)).strip()
		for latex, dom_id in self.pending:
			self.records.append(InlineMathOccurrence(
				latex, canonicalize_latex(latex), paragraph, dom_id, self.paragraph_id
			))
		self.parts = []
		self.pending = []


def build_inline_math_index(source: str | Path) -> dict[str, list[InlineMathOccurrence]]:
	text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
	parser = _Parser()
	parser.feed(text)
	parser._flush()
	output = {}
	for record in parser.records:
		output.setdefault(record.canonical, []).append(record)
	return output
