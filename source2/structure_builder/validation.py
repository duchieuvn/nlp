from collections import Counter

from models import PaperDocument


def validate_documents(
	documents: list[PaperDocument],
	expected_equations: set[tuple[str, str]],
) -> None:
	errors: list[str] = []
	paper_ids = [document.paper_id for document in documents]
	if len(paper_ids) != len(set(paper_ids)):
		errors.append("Paper IDs are not unique")

	actual_equations: set[tuple[str, str]] = set()
	for document in documents:
		section_ids = {section.section_id for section in document.sections}
		if len(section_ids) != len(document.sections):
			errors.append(f"{document.paper_id}: section IDs are not unique")
		paragraphs = {
			paragraph.paragraph_id: paragraph
			for section in document.sections
			for paragraph in section.paragraphs
		}
		sentences = {
			sentence.sentence_id: sentence
			for paragraph in paragraphs.values()
			for sentence in paragraph.sentences
		}
		equations = {equation.equation_id: equation for equation in document.equations}
		references = {
			reference.reference_id: reference
			for reference in document.cross_references
		}
		if len(paragraphs) != sum(len(section.paragraphs) for section in document.sections):
			errors.append(f"{document.paper_id}: paragraph IDs are not unique")
		if len(sentences) != sum(len(p.sentences) for p in paragraphs.values()):
			errors.append(f"{document.paper_id}: sentence IDs are not unique")
		if len(equations) != len(document.equations):
			errors.append(f"{document.paper_id}: equation IDs are not unique")
		if len(references) != len(document.cross_references):
			errors.append(f"{document.paper_id}: cross-reference IDs are not unique")

		for section in document.sections:
			if section.parent_section_id and section.parent_section_id not in section_ids:
				errors.append(
					f"{document.paper_id}: missing parent section {section.parent_section_id}"
				)
			for equation_id in section.equation_ids:
				if equation_id not in equations:
					errors.append(f"{document.paper_id}: unknown section equation {equation_id}")

		for equation in document.equations:
			actual_equations.add((document.paper_id, equation.equation_id))
			if equation.section_id and equation.section_id not in section_ids:
				errors.append(
					f"{document.paper_id}: equation {equation.equation_id} has unknown section"
				)
			for paragraph_id in (
				equation.previous_paragraph_id,
				equation.next_paragraph_id,
			):
				if paragraph_id and paragraph_id not in paragraphs:
					errors.append(
					f"{document.paper_id}: equation {equation.equation_id} has unknown paragraph"
				)

		attached_reference_ids: list[str] = []
		for paragraph in paragraphs.values():
			for equation_id in paragraph.nearby_equation_ids:
				if equation_id not in equations:
					errors.append(
					f"{document.paper_id}: paragraph has unknown equation {equation_id}"
				)
			attached_reference_ids.extend(paragraph.cross_reference_ids)
			for sentence in paragraph.sentences:
				if paragraph.text[sentence.start : sentence.end] != sentence.text:
					errors.append(
					f"{document.paper_id}: invalid sentence offsets {sentence.sentence_id}"
				)
				attached_reference_ids.extend(sentence.cross_reference_ids)

		for reference in document.cross_references:
			paragraph = paragraphs.get(reference.source_paragraph_id)
			sentence = sentences.get(reference.source_sentence_id)
			if reference.source_section_id not in section_ids or not paragraph or not sentence:
				errors.append(f"{document.paper_id}: invalid source for {reference.reference_id}")
				continue
			if paragraph.text[reference.paragraph_start : reference.paragraph_end] != reference.raw_text:
				errors.append(f"{document.paper_id}: invalid paragraph reference offsets")
			if sentence.text[reference.sentence_start : reference.sentence_end] != reference.raw_text:
				errors.append(f"{document.paper_id}: invalid sentence reference offsets")
			if any(target not in equations for target in reference.target_equation_ids):
				errors.append(f"{document.paper_id}: reference has unknown resolved target")

		attachment_counts = Counter(attached_reference_ids)
		for reference_id in references:
			if attachment_counts[reference_id] != 2:
				errors.append(
					f"{document.paper_id}: {reference_id} must attach to paragraph and sentence"
				)

	if actual_equations != expected_equations:
		missing = sorted(expected_equations - actual_equations)
		extra = sorted(actual_equations - expected_equations)
		errors.append(f"Equation registry mismatch; missing={missing[:5]}, extra={extra[:5]}")
	if errors:
		raise ValueError("Structured-document validation failed:\n- " + "\n- ".join(errors[:50]))
