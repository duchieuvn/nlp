from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .symbol_config import (
	CHECKPOINT_DIR,
	DEFAULT_RELATION_MARGIN,
	MINIMUM_REVIEWED_ACCEPTS,
	TARGET_ACCEPTED_PRECISION,
)
from .symbol_models import RELATION_LABELS


def build_balanced_review_sample(output_dir: Path, per_stratum: int = 25) -> list[dict]:
	buckets = defaultdict(list)
	for path in sorted(output_dir.glob("*.json")):
		payload = json.loads(path.read_text(encoding="utf-8"))
		for equation in payload["equations"]:
			for symbol in equation["symbols"]:
				audit = symbol.get("audit", {})
				parsed = audit.get("parsed_components", {})
				for relation in audit.get("component_relations", []):
					row = {
						"paper_id": payload["paper_id"],
						"equation_id": equation["equation_id"],
						"canonical": symbol["canonical"],
						"original_latex": parsed.get("original_latex", ""),
						"has_modifiers": bool(parsed.get("subscript") or parsed.get("superscript") or parsed.get("decorators")),
						"candidate_phrase": relation["candidate_phrase"],
						"evidence_sentence": relation["evidence_sentence"],
						"predicted_relation": relation["relation"],
						"relation_probabilities": relation["relation_probabilities"],
						"gold_relation": None,
						"review_notes": "",
					}
					buckets[relation["relation"]].append(row)
					if row["has_modifiers"]:
						buckets["modifier"].append(row)
					if " and " in relation["evidence_sentence"].casefold():
						buckets["coordinated"].append(row)
	selected = {}
	for stratum, rows in sorted(buckets.items()):
		ordered = sorted(rows, key=lambda row: hashlib.sha256(
			f"{row['paper_id']}:{row['equation_id']}:{row['canonical']}:{row['candidate_phrase']}:{stratum}".encode()
		).hexdigest())
		for row in ordered[:per_stratum]:
			key = (
				row["paper_id"], row["equation_id"], row["canonical"],
				row["candidate_phrase"], row["evidence_sentence"],
			)
			selected.setdefault(key, {**row, "review_strata": []})["review_strata"].append(stratum)
	return list(selected.values())


def write_review_sample(rows: list[dict], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps({
		"instructions": (
			"Fill gold_relation with one of the six configured relation labels. "
			"Do not change model probabilities."
		),
		"labels": list(RELATION_LABELS),
		"examples": rows,
	}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def calibrate_reviewed_file(review_file: Path, checkpoint_dir=CHECKPOINT_DIR) -> dict:
	payload = json.loads(review_file.read_text(encoding="utf-8"))
	rows = [row for row in payload["examples"] if row.get("gold_relation") in RELATION_LABELS]
	if len(rows) < MINIMUM_REVIEWED_ACCEPTS:
		raise ValueError(
			f"Need at least {MINIMUM_REVIEWED_ACCEPTS} reviewed examples; found {len(rows)}"
		)
	candidates = []
	for row in rows:
		probabilities = row["relation_probabilities"]
		predicted = max(probabilities, key=probabilities.get)
		if predicted not in {"DEFINES_COMPLETE_SYMBOL", "DEFINES_BASE"}:
			continue
		if predicted == "DEFINES_BASE" and row.get("has_modifiers"):
			continue
		ordered = sorted(probabilities.values(), reverse=True)
		margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
		candidates.append((ordered[0], margin, predicted == row["gold_relation"]))
	best = None
	for threshold in sorted({score for score, _, _ in candidates}):
		accepted = [item for item in candidates if item[0] >= threshold and item[1] >= DEFAULT_RELATION_MARGIN]
		if len(accepted) < MINIMUM_REVIEWED_ACCEPTS:
			continue
		precision = sum(item[2] for item in accepted) / len(accepted)
		if precision >= TARGET_ACCEPTED_PRECISION:
			candidate = (len(accepted), threshold, precision)
			if best is None or candidate[0] > best[0]:
				best = candidate
	if best is None:
		raise ValueError(
			f"No threshold reached reviewed precision {TARGET_ACCEPTED_PRECISION:.2f} "
			f"with at least {MINIMUM_REVIEWED_ACCEPTS} accepted examples"
		)
	accepted, threshold, precision = best
	calibration = {
		"relation_threshold": threshold,
		"relation_margin": DEFAULT_RELATION_MARGIN,
		"labels": list(RELATION_LABELS),
		"calibration_scope": "human-reviewed balanced symbol-relation validation sample",
		"reviewed_examples": len(rows),
		"accepted_examples": accepted,
		"accepted_precision": precision,
		"target_precision": TARGET_ACCEPTED_PRECISION,
	}
	(checkpoint_dir / "inference_config.json").write_text(
		json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
	)
	return calibration
