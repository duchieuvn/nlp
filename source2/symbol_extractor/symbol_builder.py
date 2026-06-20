from symbol_extractor import extract_symbols


def build_paper_symbols(paper_id: str, equations: dict) -> dict:
	return {
		"paper_id": paper_id,
		"equations": [
			{
				"equation_id": equation_id,
				"latex": entry["equation"],
				"symbols": [
					symbol.to_dict()
					for symbol in extract_symbols(entry["equation"])
				],
			}
			for equation_id, entry in equations.items()
		],
	}
