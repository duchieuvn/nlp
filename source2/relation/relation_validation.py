from relation_config import VALID_DESCRIPTIONS, VALID_GRADES


def validate_relation_payload(payload: dict) -> None:
	equation_ids = [equation["equation_id"] for equation in payload["equations"]]
	if len(equation_ids) != len(set(equation_ids)):
		raise ValueError(f"Duplicate equation IDs in {payload['paper_id']}")
	for equation in payload["equations"]:
		expected = set(equation_ids) - {equation["equation_id"]}
		actual = set(equation["relations"])
		if actual != expected:
			raise ValueError(
				f"Incomplete relations for {payload['paper_id']} equation "
				f"{equation['equation_id']}: expected {expected}, got {actual}"
			)
		for target_id, relation in equation["relations"].items():
			if relation["grade"] not in VALID_GRADES:
				raise ValueError(f"Invalid grade for relation to {target_id}")
			if relation["description"] not in VALID_DESCRIPTIONS:
				raise ValueError(f"Invalid description for relation to {target_id}")
			if relation["grade"] == "none" and relation["description"]:
				raise ValueError("None relation must have an empty description")
