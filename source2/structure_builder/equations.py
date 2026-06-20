from dataclasses import dataclass
import re
from typing import Any

from bs4 import BeautifulSoup

from html_document import ParsedHtml
from models import EquationRecord
from text import normalize_latex


EQUATION_CLASSES = {"ltx_equation", "ltx_equationgroup"}


def is_equation_table(node) -> bool:
	return bool(
		node
		and node.name == "table"
		and EQUATION_CLASSES & set(node.get("class") or [])
	)


def equation_table_for(node):
	if is_equation_table(node):
		return node
	return node.find_parent(is_equation_table) if node else None


def annotation_ids(annotation_html_list: list[str]) -> list[str]:
	result = []
	for annotation_html in annotation_html_list:
		annotation = BeautifulSoup(annotation_html, "html.parser").find("annotation")
		if annotation and annotation.get("id"):
			result.append(annotation["id"])
	return list(dict.fromkeys(result))


def _context_audit(entry: dict[str, Any]) -> dict[str, Any]:
	for item in reversed(entry.get("audit-trail", [])):
		if isinstance(item, dict) and isinstance(item.get("context_extraction"), dict):
			return item["context_extraction"]
	return {}


def visible_equation_labels(table) -> list[str]:
	labels = []
	for node in table.find_all(
		class_=lambda classes: classes
		and (
			"ltx_eqn_eqno" in classes
			if isinstance(classes, list)
			else "ltx_eqn_eqno" in classes
		)
	):
		label = re.sub(r"\s+", "", node.get_text(" ", strip=True))
		label = label.strip("()")
		if label:
			labels.append(label)
	return list(dict.fromkeys(labels))


def table_latex_forms(table) -> set[str]:
	raw_parts = []
	for annotation in table.find_all("annotation"):
		if annotation.get("encoding") == "application/x-tex":
			raw = annotation.get_text()
			if raw.strip():
				raw_parts.append(raw)
	forms = {normalize_latex(part) for part in raw_parts if normalize_latex(part)}
	if raw_parts:
		joined = normalize_latex(" ".join(raw_parts))
		if joined:
			forms.add(joined)
	for math in table.find_all("math"):
		alttext = math.get("alttext")
		if alttext:
			normalized = normalize_latex(alttext)
			if normalized:
				forms.add(normalized)
	return forms


@dataclass(frozen=True)
class EquationResolution:
	record: EquationRecord
	table: object | None


class EquationResolver:
	def __init__(self, parsed: ParsedHtml) -> None:
		self.parsed = parsed
		self.tables = [
			table
			for table in parsed.soup.find_all("table")
			if is_equation_table(table)
		]
		self.labels: dict[str, list[object]] = {}
		self.formulas: dict[int, set[str]] = {}
		for table in self.tables:
			for label in visible_equation_labels(table):
				self.labels.setdefault(label, []).append(table)
			self.formulas[id(table)] = table_latex_forms(table)

	def _table_from_dom_id(self, dom_id: str | None):
		if not dom_id:
			return None
		return equation_table_for(self.parsed.soup.find(id=dom_id))

	def _exact_formula_tables(self, latex: str, candidates=None) -> list[object]:
		normalized = normalize_latex(latex)
		search_tables = self.tables if candidates is None else candidates
		return [
			table
			for table in search_tables
			if normalized and normalized in self.formulas[id(table)]
		]

	def resolve(
		self,
		equation_id: str,
		entry: dict[str, Any],
		annotation_html_list: list[str],
	) -> EquationResolution:
		ids = annotation_ids(annotation_html_list)
		audit = _context_audit(entry)
		matched_annotation_id = audit.get("matched_annotation_id")
		if isinstance(matched_annotation_id, str):
			ids.append(matched_annotation_id)
		ids = list(dict.fromkeys(ids))

		table = None
		match_method = "unresolved"
		for dom_id in ids:
			table = self._table_from_dom_id(dom_id)
			if table is not None:
				match_method = "annotation_id"
				break

		if table is None:
			audit_anchor = audit.get("anchor_id")
			if isinstance(audit_anchor, str):
				table = self._table_from_dom_id(audit_anchor)
				if table is not None:
					match_method = "audit_anchor_id"

		if table is None:
			label_candidates = self.labels.get(equation_id, [])
			if len(label_candidates) == 1:
				table = label_candidates[0]
				match_method = "visible_label"
			elif len(label_candidates) > 1:
				exact = self._exact_formula_tables(entry.get("equation", ""), label_candidates)
				if len(exact) == 1:
					table = exact[0]
					match_method = "visible_label_and_latex"

		if table is None:
			exact = self._exact_formula_tables(entry.get("equation", ""))
			if len(exact) == 1:
				table = exact[0]
				match_method = "exact_latex"

		section_id = self.parsed.section_id_for(table) if table is not None else None
		context = entry.get("surrounding_text", {})
		record = EquationRecord(
			equation_id=equation_id,
			latex=entry.get("equation", ""),
			section_id=section_id,
			document_order=(
				self.parsed.node_positions.get(id(table)) if table is not None else None
			),
			anchor_id=table.get("id") if table is not None else None,
			annotation_ids=ids,
			match_method=match_method,
			previous_paragraph_id=None,
			next_paragraph_id=None,
			legacy_context_before=context.get("before", ""),
			legacy_context_after=context.get("after", ""),
		)
		return EquationResolution(record=record, table=table)
