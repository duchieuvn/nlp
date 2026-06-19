"""One-click, zero-label MathBERTA baseline for equation-name extraction.

Run this file from the IDE. On its first run it installs missing Python
packages and downloads the model. It never reads the reviewed NER dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import math
import re
import subprocess
import sys
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "3_equations.json"
OUTPUT_FILE = PROJECT_DIR / "data" / "4b_mathbert_baseline.json"
MODEL_NAME = "witiko/mathberta"
EQUATION_MARKER = "[EQUATION]"
AUTO_INSTALL_DEPENDENCIES = True
MAX_CONTEXT_CHARACTERS = 1800
MAX_EQUATION_CHARACTERS = 500
MAX_MODEL_TOKENS = 256
MIN_CONFIDENCE = 0.45

SCIENCE_HEADS = {
    "condition", "constraint", "distribution", "equation", "expression",
    "formula", "function", "hamiltonian", "identity", "inequality",
    "lagrangian", "law", "matrix", "operator", "relation", "state",
    "theorem", "value",
}
LEADING_DROP_WORDS = {"a", "an", "and", "our", "the", "their", "this", "where"}
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
REFERENCE_PATTERN = re.compile(r"\b(?:eq|fig|sec|app|ref)s?\.?\s*\(?\s*\d+", re.I)


@dataclass(frozen=True)
class Candidate:
    text: str
    start: int
    end: int
    distance: int
    cue_bonus: float = 0.0


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def ensure_dependencies() -> None:
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if not AUTO_INSTALL_DEPENDENCIES:
        raise RuntimeError(f"Missing packages: {', '.join(missing)}")
    print("Installing the MathBERT runtime. This is required only once...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torch", "transformers"]
    )


def _clean_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip(" ,;:."))
    right = len(raw.rstrip(" ,;:."))
    start += left
    end = start + max(0, right - left)
    words = list(WORD_PATTERN.finditer(text, start, end))
    while words and words[0].group().lower() in LEADING_DROP_WORDS:
        start = words[0].end()
        while start < end and text[start].isspace():
            start += 1
        words.pop(0)
    return text[start:end].strip(), start, end


def _valid_candidate(text: str) -> bool:
    words = WORD_PATTERN.findall(text)
    return (
        1 <= len(words) <= 12
        and len(text) >= 3
        and not REFERENCE_PATTERN.search(text)
        and not any(character in text for character in "[]{}")
    )


def candidate_phrases(window: str) -> list[Candidate]:
    """Return literal natural-language spans before the equation marker."""
    marker = window.find(EQUATION_MARKER)
    if marker < 0:
        marker = len(window)
    region_start = max(0, marker - MAX_CONTEXT_CHARACTERS)
    before = window[region_start:marker]
    words = list(WORD_PATTERN.finditer(before))
    candidates: dict[tuple[int, int], Candidate] = {}

    cue_pattern = re.compile(
        r"(?P<phrase>[A-Za-z][^.!?;:\n]{2,140}?)\s+"
        r"(?:is|are|can\s+be|may\s+be)\s+"
        r"(?:given|written|expressed|defined|represented|implemented)(?:\s+by|\s+as)?\s*$",
        re.I,
    )
    cue_match = cue_pattern.search(before)
    if cue_match:
        raw_start, raw_end = cue_match.span("phrase")
        sentence_start = max(
            before.rfind(".", 0, raw_end),
            before.rfind(":", 0, raw_end),
            before.rfind(";", 0, raw_end),
        ) + 1
        raw_start = max(raw_start, sentence_start)
        text, start, end = _clean_span(before, raw_start, raw_end)
        if _valid_candidate(text):
            absolute_start = region_start + start
            absolute_end = region_start + end
            candidates[(absolute_start, absolute_end)] = Candidate(
                text, absolute_start, absolute_end, marker - absolute_end, 0.18
            )

    for head_index, word in enumerate(words):
        if word.group().lower() not in SCIENCE_HEADS:
            continue
        for left_count in range(5):
            left_index = max(0, head_index - left_count)
            for right_count in range(5):
                right_index = min(len(words) - 1, head_index + right_count)
                if right_index - left_index + 1 > 9:
                    continue
                start = words[left_index].start()
                end = words[right_index].end()
                if any(mark in before[start:end] for mark in ".;:?!"):
                    continue
                text, clean_start, clean_end = _clean_span(before, start, end)
                if not _valid_candidate(text):
                    continue
                absolute_start = region_start + clean_start
                absolute_end = region_start + clean_end
                candidates[(absolute_start, absolute_end)] = Candidate(
                    text, absolute_start, absolute_end, marker - absolute_end
                )

    return sorted(candidates.values(), key=lambda item: (item.distance, len(item.text)))[:40]


def model_context(window: str) -> str:
    marker = window.find(EQUATION_MARKER)
    if marker < 0:
        return window[:MAX_CONTEXT_CHARACTERS]
    before = window[max(0, marker - 1200):marker]
    after_start = marker + len(EQUATION_MARKER)
    after = window[after_start:after_start + 600]
    return normalize_text(f"{before} {EQUATION_MARKER} {after}")


def load_mathbert() -> tuple[Any, Any, Any, str]:
    ensure_dependencies()
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, torch, device


def embed_texts(
    texts: list[str], tokenizer: Any, model: Any, torch: Any, device: str
) -> Any:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_MODEL_TOKENS,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    embeddings = summed / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def predict_meaning(
    window: str,
    equation: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
) -> dict[str, Any]:
    candidates = candidate_phrases(window)
    if not candidates:
        return {
            "meaning": "", "candidate": "", "confidence": 0.0,
            "status": "no_candidate", "start": None, "end": None,
        }

    context = model_context(window)
    equation_text = normalize_text(equation)[:MAX_EQUATION_CHARACTERS]
    reference_texts = [f"scientific context: {context}"]
    if equation_text:
        reference_texts.append(f"mathematical equation: {equation_text}")
    candidate_texts = [f"equation name: {candidate.text}" for candidate in candidates]
    embeddings = embed_texts(reference_texts + candidate_texts, tokenizer, model, torch, device)
    context_embedding = embeddings[0]
    equation_embedding = embeddings[1] if equation_text else context_embedding
    candidate_embeddings = embeddings[len(reference_texts):]

    ranked = []
    for candidate, embedding in zip(candidates, candidate_embeddings):
        semantic = 0.65 * float(torch.dot(embedding, context_embedding))
        semantic += 0.35 * float(torch.dot(embedding, equation_embedding))
        proximity = 0.10 * math.exp(-candidate.distance / 250)
        length_penalty = max(0, len(WORD_PATTERN.findall(candidate.text)) - 8) * 0.01
        score = semantic + proximity + candidate.cue_bonus - length_penalty
        ranked.append((score, candidate))

    score, best = max(ranked, key=lambda item: item[0])
    confidence = max(0.0, min(1.0, (score + 1.0) / 2.0))
    accepted = confidence >= MIN_CONFIDENCE
    return {
        "meaning": best.text if accepted else "",
        "candidate": best.text,
        "confidence": confidence,
        "status": "accepted" if accepted else "rejected_low_confidence",
        "start": best.start,
        "end": best.end,
    }


def run_baseline() -> tuple[int, int]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    tokenizer, model, torch, device = load_mathbert()
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    total = 0
    filled = 0
    for paper_equations in data.values():
        for entry in paper_equations.values():
            total += 1
            result = predict_meaning(
                entry.get("surrounding_text", {}).get("window", ""),
                entry.get("equation", ""),
                tokenizer,
                model,
                torch,
                device,
            )
            entry["meaning"] = result["meaning"]
            filled += bool(result["meaning"])
            entry.setdefault("audit-trail", []).append({
                "meaning_extraction": {
                    "method": "Zero-label MathBERTA embedding candidate ranking",
                    "model": MODEL_NAME,
                    "strategy": "mathberta_embedding_baseline",
                    **result,
                }
            })
            print(f"Processed {total}: {result['meaning'] or '[blank]'}")

    OUTPUT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return total, filled


def main() -> None:
    try:
        total, filled = run_baseline()
    except Exception as exc:
        print(f"MathBERT baseline failed: {exc}", file=sys.stderr)
        raise
    print(f"Done. Filled {filled} of {total} meanings.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
