from bisect import bisect_left, bisect_right
import json
from pathlib import Path
from typing import Any

from config import PROJECT_DIR
from equations import EquationResolver
from html_document import ParsedHtml, parse_html_document
from models import CrossReferenceRecord, PaperDocument, SCHEMA_VERSION
from references import iter_explicit_references
from sentences import SpacySentenceSegmenter
from validation import validate_documents


def _relative_path(path: Path) -> str:
	try:
		return str(path.relative_to(PROJECT_DIR))
	except ValueError:
		return str(path)


def _attach_equation_neighbors(parsed: ParsedHtml, resolutions) -> None:
	paragraphs = sorted(
		parsed.paragraphs_by_id.values(), key=lambda paragraph: paragraph.document_order
	)
	orders = [paragraph.document_order for paragraph in paragraphs]
	for resolution in resolutions:
		record = resolution.record
		if record.document_order is None:
			continue
		previous_index = bisect_left(orders, record.document_order) - 1
		next_index = bisect_right(orders, record.document_order)
		if previous_index >= 0:
			previous = paragraphs[previous_index]
			record.previous_paragraph_id = previous.paragraph_id
			if record.equation_id not in previous.nearby_equation_ids:
				previous.nearby_equation_ids.append(record.equation_id)
		if next_index < len(paragraphs):
			next_paragraph = paragraphs[next_index]
			record.next_paragraph_id = next_paragraph.paragraph_id
			if record.equation_id not in next_paragraph.nearby_equation_ids:
				next_paragraph.nearby_equation_ids.append(record.equation_id)


def _extract_cross_references(
	paper_id: str,
	parsed: ParsedHtml,
	target_equation_ids: set[str],
) -> list[CrossReferenceRecord]:
	references = []
	for section in parsed.ordered_sections():
		for paragraph in section.paragraphs:
			for sentence in paragraph.sentences:
				for match, reference_type, labels in iter_explicit_references(sentence.text):
					reference_id = f"{paper_id}:xref:{len(references) + 1}"
					resolved = list(dict.fromkeys(
						label for label in labels if label in target_equation_ids
					))
					unresolved = list(dict.fromkeys(
						label for label in labels if label not in target_equation_ids
					))
					reference = CrossReferenceRecord(
						reference_id=reference_id,
						raw_text=match.group(0),
						reference_type=reference_type,
						source_section_id=section.section_id,
						source_paragraph_id=paragraph.paragraph_id,
						source_sentence_id=sentence.sentence_id,
						paragraph_start=sentence.start + match.start(),
						paragraph_end=sentence.start + match.end(),
						sentence_start=match.start(),
						sentence_end=match.end(),
						target_equation_ids=resolved,
						unresolved_labels=unresolved,
					)
					references.append(reference)
					paragraph.cross_reference_ids.append(reference_id)
					sentence.cross_reference_ids.append(reference_id)
	return references


def build_paper_document(
	paper_id: str,
	equations: dict[str, dict[str, Any]],
	annotations: dict[str, list[str]],
	html_file: Path,
	segmenter: SpacySentenceSegmenter,
) -> PaperDocument:
	parsed = parse_html_document(
		html_file.read_text(encoding="utf-8", errors="ignore"), segmenter
	)
	resolver = EquationResolver(parsed)
	resolutions = [
		resolver.resolve(equation_id, entry, annotations.get(equation_id, []))
		for equation_id, entry in equations.items()
	]
	_attach_equation_neighbors(parsed, resolutions)
	for resolution in resolutions:
		section_id = resolution.record.section_id
		if section_id:
			parsed.sections_by_id[section_id].equation_ids.append(
				resolution.record.equation_id
			)
	cross_references = _extract_cross_references(
		paper_id, parsed, set(equations)
	)
	return PaperDocument(
		paper_id=paper_id,
		title=parsed.title,
		html_source=_relative_path(html_file),
		source_status="html",
		sections=parsed.ordered_sections(),
		equations=[resolution.record for resolution in resolutions],
		cross_references=cross_references,
	)


def build_corpus(
	annotations_file: Path,
	equations_file: Path,
	html_dir: Path,
) -> dict[str, Any]:
	annotations = json.loads(annotations_file.read_text(encoding="utf-8"))
	equations = json.loads(equations_file.read_text(encoding="utf-8"))
	empty_papers = [
		{
			"paper_id": paper_id,
			"reason": "no_target_equations",
			"html_available": (html_dir / f"{paper_id}.html").exists(),
		}
		for paper_id, entries in equations.items()
		if not entries
	]
	nonempty = {paper_id: entries for paper_id, entries in equations.items() if entries}
	missing_html = [
		paper_id
		for paper_id in nonempty
		if not (html_dir / f"{paper_id}.html").exists()
	]
	if missing_html:
		raise FileNotFoundError(
			"Target-equation papers are missing HTML: " + ", ".join(missing_html)
		)

	segmenter = SpacySentenceSegmenter()
	documents = [
		build_paper_document(
			paper_id,
			entries,
			annotations.get(paper_id, {}),
			html_dir / f"{paper_id}.html",
			segmenter,
		)
		for paper_id, entries in nonempty.items()
	]
	expected_equations = {
		(paper_id, equation_id)
		for paper_id, entries in nonempty.items()
		for equation_id in entries
	}
	validate_documents(documents, expected_equations)

	unresolved_equations = [
		{"paper_id": document.paper_id, "equation_id": equation.equation_id}
		for document in documents
		for equation in document.equations
		if equation.match_method == "unresolved"
	]
	all_references = [
		reference
		for document in documents
		for reference in document.cross_references
	]
	report = {
		"input_paper_count": len(equations),
		"built_paper_count": len(documents),
		"skipped_empty_paper_count": len(empty_papers),
		"target_equation_count": len(expected_equations),
		"resolved_equation_count": len(expected_equations) - len(unresolved_equations),
		"unresolved_equation_count": len(unresolved_equations),
		"paragraph_count": sum(
			len(section.paragraphs)
			for document in documents
			for section in document.sections
		),
		"sentence_count": sum(
			len(paragraph.sentences)
			for document in documents
			for section in document.sections
			for paragraph in section.paragraphs
		),
		"cross_reference_count": len(all_references),
		"fully_resolved_cross_reference_count": sum(
			not reference.unresolved_labels for reference in all_references
		),
		"empty_papers": empty_papers,
		"unresolved_equations": unresolved_equations,
	}
	return {
		"schema_version": SCHEMA_VERSION,
		"sources": {
			"annotations": _relative_path(annotations_file),
			"equations": _relative_path(equations_file),
			"html_directory": _relative_path(html_dir),
		},
		"papers": [document.to_dict() for document in documents],
		"build_report": report,
	}


def write_papers(corpus: dict[str, Any], output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	for paper in corpus["papers"]:
		paper_id = paper["paper_id"]
		if not isinstance(paper_id, str) or Path(paper_id).name != paper_id:
			raise ValueError(f"Invalid paper ID for output filename: {paper_id!r}")
		output_file = output_dir / f"{paper_id}.json"
		temporary_file = output_file.with_suffix(".json.tmp")
		temporary_file.write_text(
			json.dumps(paper, ensure_ascii=False, indent=2) + "\n",
			encoding="utf-8",
		)
		temporary_file.replace(output_file)
