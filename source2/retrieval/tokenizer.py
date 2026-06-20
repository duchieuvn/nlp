import re
import unicodedata


GREEK_NAMES = {
	"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
	"ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
	"ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
	"ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
	"ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
	"φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
	"Γ": "gamma", "Δ": "delta", "Θ": "theta", "Λ": "lambda",
	"Ξ": "xi", "Π": "pi", "Σ": "sigma", "Υ": "upsilon",
	"Φ": "phi", "Ψ": "psi", "Ω": "omega",
}

TOKEN_PATTERN = re.compile(
	r"\\[A-Za-z]+(?:_\{[^{}]+\}|_[A-Za-z0-9]+)?"
	r"|[^\W\d_][\w]*(?:_[\w]+)*"
	r"|\d+(?:\.\d+)?",
	re.UNICODE,
)


def _normalize_token(token: str) -> str:
	token = unicodedata.normalize("NFKC", token)
	if token in GREEK_NAMES:
		return GREEK_NAMES[token]
	if token.startswith("\\"):
		token = token[1:]
	token = token.replace("_{", "_").replace("}", "")
	return token.casefold()


def tokenize(text: str) -> list[str]:
	tokens = []
	for match in TOKEN_PATTERN.finditer(text):
		token = _normalize_token(match.group(0))
		if not token:
			continue
		tokens.append(token)
		if "_" in token:
			tokens.extend(part for part in token.split("_") if part)
	return tokens
