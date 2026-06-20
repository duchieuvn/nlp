MEANING_CUES = (
	"defines",
	"describes",
	"represents",
	"gives",
	"called",
	"known as",
)


def build_equation_meaning_query(equation: dict) -> str:
	equation_id = equation["equation_id"]
	symbols = [symbol["canonical"] for symbol in equation.get("symbols", [])]
	terms = [
		"Eq",
		equation_id,
		"Equation",
		equation_id,
		*MEANING_CUES,
		*dict.fromkeys(symbols),
	]
	return " ".join(terms)
