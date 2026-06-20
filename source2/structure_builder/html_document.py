from collections import defaultdict
from dataclasses import dataclass
import re

from bs4 import BeautifulSoup

from models import ParagraphRecord, SectionRecord
from sentences import SpacySentenceSegmenter
from text import clean_text_node, normalize_text


SECTION_CLASSES = {
	"ltx_section": (1, "section"),
	"ltx_subsection": (2, "subsection"),
	"ltx_subsubsection": (3, "subsubsection"),
	"ltx_paragraph": (4, "paragraph"),
	"ltx_appendix": (1, "appendix"),
}
EXCLUDED_SECTION_CLASSES = {"ltx_bibliography"}
EXCLUDED_ANCESTOR_TAGS = {"figure", "table", "nav", "footer", "header"}


def _classes(node) -> set[str]:
	return set(node.get("class") or [])


def _section_spec(node) -> tuple[int, str] | None:
	for class_name, specification in SECTION_CLASSES.items():
		if class_name in _classes(node):
			return specification
	return None


def _is_bibliography(node) -> bool:
	return bool(_classes(node) & EXCLUDED_SECTION_CLASSES)


def _heading_text(section) -> str:
	heading = section.find(re.compile(r"^h[1-6]$"), recursive=False)
	if heading is None:
		heading = section.find(re.compile(r"^h[1-6]$"))
	return normalize_text(heading.get_text(" ", strip=True)) if heading else ""


def _document_title(soup: BeautifulSoup) -> str:
	title = soup.find(class_="ltx_title_document") or soup.find("h1")
	if title:
		return normalize_text(title.get_text(" ", strip=True))
	if soup.title:
		return normalize_text(soup.title.get_text(" ", strip=True))
	return ""


def _paragraph_is_included(paragraph) -> bool:
	for ancestor in paragraph.parents:
		if getattr(ancestor, "name", None) in EXCLUDED_ANCESTOR_TAGS:
			return False
		if getattr(ancestor, "attrs", None) and _is_bibliography(ancestor):
			return False
	return True


@dataclass
class ParsedHtml:
	soup: BeautifulSoup
	title: str
	sections_by_id: dict[str, SectionRecord]
	section_tags: dict[int, str]
	section_positions: dict[str, int]
	node_positions: dict[int, int]
	paragraph_nodes: dict[str, object]
	paragraphs_by_id: dict[str, ParagraphRecord]
	_used_section_ids: set[str]

	def _synthetic_section(self, kind: str, position: int) -> str:
		section_id = f"__{kind}__"
		if section_id not in self.sections_by_id:
			title = "Abstract" if kind == "abstract" else "Main matter"
			self.sections_by_id[section_id] = SectionRecord(
				section_id=section_id,
				parent_section_id=None,
				order=0,
				level=0 if kind == "abstract" else 1,
				kind=kind,
				title=title,
				synthetic=True,
			)
			self.section_positions[section_id] = position
			self._used_section_ids.add(section_id)
		else:
			self.section_positions[section_id] = min(
				self.section_positions[section_id], position
			)
		return section_id

	def section_id_for(self, node) -> str:
		section = node.find_parent("section")
		while section is not None:
			section_id = self.section_tags.get(id(section))
			if section_id:
				return section_id
			section = section.find_parent("section")

		position = self.node_positions.get(id(node), 0)
		abstract = node.find_parent(class_="ltx_abstract")
		if abstract is not None:
			return self._synthetic_section("abstract", position)
		return self._synthetic_section("main", position)

	def ordered_sections(self) -> list[SectionRecord]:
		sections = sorted(
			self.sections_by_id.values(),
			key=lambda section: (
				self.section_positions.get(section.section_id, 0),
				section.section_id,
			),
		)
		for order, section in enumerate(sections, start=1):
			section.order = order
			section.paragraphs.sort(key=lambda paragraph: paragraph.document_order)
			for paragraph_order, paragraph in enumerate(section.paragraphs, start=1):
				paragraph.order = paragraph_order
		return sections


def parse_html_document(
	html: str,
	segmenter: SpacySentenceSegmenter,
) -> ParsedHtml:
	soup = BeautifulSoup(html, "html.parser")
	all_nodes = soup.find_all(True)
	node_positions = {id(node): position for position, node in enumerate(all_nodes)}
	sections_by_id: dict[str, SectionRecord] = {}
	section_tags: dict[int, str] = {}
	section_positions: dict[str, int] = {}
	used_section_ids: set[str] = set()
	section_counter = 0

	for section in soup.find_all("section"):
		if _is_bibliography(section):
			continue
		specification = _section_spec(section)
		if specification is None:
			continue
		section_counter += 1
		base_id = section.get("id") or f"section:{section_counter}"
		section_id = base_id
		suffix = 2
		while section_id in used_section_ids:
			section_id = f"{base_id}:{suffix}"
			suffix += 1
		used_section_ids.add(section_id)
		level, kind = specification
		parent = section.find_parent("section")
		parent_section_id = None
		while parent is not None:
			parent_section_id = section_tags.get(id(parent))
			if parent_section_id:
				break
			parent = parent.find_parent("section")
		sections_by_id[section_id] = SectionRecord(
			section_id=section_id,
			parent_section_id=parent_section_id,
			order=0,
			level=level,
			kind=kind,
			title=_heading_text(section),
			synthetic=False,
		)
		section_tags[id(section)] = section_id
		section_positions[section_id] = node_positions[id(section)]

	parsed = ParsedHtml(
		soup=soup,
		title=_document_title(soup),
		sections_by_id=sections_by_id,
		section_tags=section_tags,
		section_positions=section_positions,
		node_positions=node_positions,
		paragraph_nodes={},
		paragraphs_by_id={},
		_used_section_ids=used_section_ids,
	)
	paragraph_id_counts: defaultdict[str, int] = defaultdict(int)

	for paragraph in soup.find_all("p", class_="ltx_p"):
		if not _paragraph_is_included(paragraph):
			continue
		text = clean_text_node(paragraph)
		if not text:
			continue
		section_id = parsed.section_id_for(paragraph)
		base_id = paragraph.get("id") or f"{section_id}.p"
		paragraph_id_counts[base_id] += 1
		paragraph_id = base_id
		if paragraph_id_counts[base_id] > 1:
			paragraph_id = f"{base_id}:{paragraph_id_counts[base_id]}"
		record = ParagraphRecord(
			paragraph_id=paragraph_id,
			order=0,
			document_order=node_positions[id(paragraph)],
			text=text,
			sentences=segmenter.records(paragraph_id, text),
		)
		parsed.sections_by_id[section_id].paragraphs.append(record)
		parsed.paragraph_nodes[paragraph_id] = paragraph
		parsed.paragraphs_by_id[paragraph_id] = record

	return parsed
