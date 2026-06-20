"""Extract equation meanings from equation_windows.yaml using Claude.

Rule (from rules.md): meanings must completely exist in the window
(only extraction, not generated).
"""

from __future__ import annotations

import re
import sys
import yaml
from pathlib import Path

import anthropic

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "equation_windows.yaml"
OUTPUT_FILE = PROJECT_DIR / "data" / "llm_groundtruth" / "equation_meanings.yaml"
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are an expert in scientific literature. "
    "Extract the name or meaning of a mathematical equation from the surrounding window text. "
    "Rules:\n"
    "1. The extracted phrase MUST appear verbatim (word-for-word) in the window text.\n"
    "2. Do NOT paraphrase, summarize, or generate new descriptions.\n"
    "3. Return a concise noun phrase (2–8 words) that names or describes what the equation "
    "represents (e.g. 'covariance matrix of the TMST state', 'diffusion constant').\n"
    "4. If no suitable phrase can be found in the window, reply with exactly: NO_MEANING_FOUND\n"
    "5. Reply with ONLY the extracted phrase or NO_MEANING_FOUND — nothing else."
)

USER_TEMPLATE = (
    "Window text (the equation appears where [EQUATION] is shown):\n\n"
    "{window}\n\n"
    "Extract the meaning/name of [EQUATION] as a verbatim phrase from the text above."
)


def window_with_marker(window: str) -> str:
    """Replace <starteqn>…<endeqn> with [EQUATION]."""
    return re.sub(r"<starteqn>.*?<endeqn>", "[EQUATION]", window, flags=re.DOTALL)


def phrase_in_window(phrase: str, window: str) -> bool:
    """Return True if phrase occurs verbatim (case-insensitive) in window."""
    return phrase.lower() in window.lower()


def extract_meaning(client: anthropic.Anthropic, window: str) -> str:
    display_window = window_with_marker(window)
    response = client.messages.create(
        model=MODEL,
        max_tokens=120,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(window=display_window)}],
    )
    text = response.content[0].text.strip()
    return "" if text == "NO_MEANING_FOUND" else text


def main() -> None:
    if not INPUT_FILE.exists():
        print(f"Input not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    with open(INPUT_FILE, encoding="utf-8") as fh:
        equations: list[dict] = yaml.safe_load(fh)

    results: list[dict] = []
    total = filled = violations = 0

    for entry in equations:
        total += 1
        paper_id = entry["paper_id"]
        eq_id = entry["equation_id"]
        window = entry["window"]

        print(f"[{total}/{len(equations)}] {paper_id} eq {eq_id} … ", end="", flush=True)

        meaning = extract_meaning(client, window)

        if meaning and not phrase_in_window(meaning, window):
            print(f"RULE VIOLATION – '{meaning}' not in window; blanked")
            violations += 1
            meaning = ""
        elif meaning:
            print(f"'{meaning}'")
            filled += 1
        else:
            print("(no meaning found)")

        results.append({
            "paper_id": paper_id,
            "equation_id": eq_id,
            "equation": entry["equation"],
            "meaning": meaning or None,
            "window": window,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        yaml.dump(results, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(
        f"\nDone. {total} equations — {filled} meanings extracted, "
        f"{total - filled - violations} blank, {violations} rule violations blanked."
    )
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
