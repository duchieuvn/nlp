GREEK_COMMANDS = {
	"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
	"epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
	"theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
	"lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
	"omicron": "ο", "rho": "ρ", "varrho": "ϱ", "sigma": "σ",
	"tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "ϕ",
	"chi": "χ", "psi": "ψ", "omega": "ω",
	"Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
	"Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
	"Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

DECORATORS = {
	"bar": "bar",
	"ddot": "double_dot",
	"dot": "dot",
	"hat": "hat",
	"overline": "bar",
	"tilde": "tilde",
	"vec": "vec",
	"widehat": "hat",
	"widetilde": "tilde",
}

FORMATTING_COMMANDS = {
	"bf", "bm", "boldsymbol", "cal", "mathbf", "mathbfcal", "mathbb",
	"mathcal", "mathrm", "mathsf", "rm",
}

DROP_GROUP_COMMANDS = {
	"label", "mbox", "operatorname", "tag", "text", "textrm",
}

NON_SYMBOL_COMMANDS = {
	"Big", "Bigg", "big", "bigg", "begin", "bullet", "cdot", "cdots",
	"circ", "cos", "cosh", "dagger", "dots", "ell", "end", "equiv",
	"exp", "frac", "geq", "hbar", "in", "infty", "int", "kern",
	"langle", "ldots", "left", "leq", "log", "max", "min", "nabla",
	"neq", "nobreak", "otimes", "partial", "pi", "pm", "prod", "quad",
	"rangle", "right", "rightarrow", "sim", "sin", "sinh", "sqrt", "sum",
	"times", "underbrace", "vdots",
}

LIKELY_INDICES = {"d", "e", "i", "j", "k", "l", "m", "n"}


def read_command(text: str, position: int) -> tuple[str, int]:
	end = position + 1
	while end < len(text) and text[end].isalpha():
		end += 1
	return text[position + 1:end], end


def read_group(text: str, position: int) -> tuple[str, int] | None:
	if position >= len(text) or text[position] != "{":
		return None
	depth = 0
	for index in range(position, len(text)):
		if text[index] == "{":
			depth += 1
		elif text[index] == "}":
			depth -= 1
			if depth == 0:
				return text[position + 1:index], index + 1
	return None


def skip_space(text: str, position: int) -> int:
	while position < len(text) and text[position].isspace():
		position += 1
	return position
