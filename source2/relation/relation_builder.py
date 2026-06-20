from relation_classifier import classify_relation
from relation_context import build_equation_contexts
from relation_validation import validate_relation_payload


def build_paper_relations(document: dict, symbol_payload: dict) -> dict:
	if document["paper_id"] != symbol_payload["paper_id"]:
		raise ValueError("Structured document and symbol registry do not match")
	contexts = build_equation_contexts(document)
	symbols = {
		equation["equation_id"]: {
			symbol["canonical"] for symbol in equation.get("symbols", [])
		}
		for equation in symbol_payload["equations"]
	}
	if set(contexts) != set(symbols):
		raise ValueError(f"Equation mismatch for paper {document['paper_id']}")
	section_positions = {
		section["section_id"]: index
		for index, section in enumerate(document["sections"])
	}
	equations = []
	for source_id, source in contexts.items():
		relations = {}
		for target_id, target in contexts.items():
			if source_id == target_id:
				continue
			relations[target_id] = classify_relation(
				source,
				target,
				symbols[source_id],
				symbols[target_id],
				document.get("cross_references", []),
				section_positions,
			)
		equations.append({
			"equation_id": source_id,
			"relations": relations,
		})
	payload = {
		"paper_id": document["paper_id"],
		"equations": equations,
	}
	validate_relation_payload(payload)
	return payload
