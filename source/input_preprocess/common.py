from pathlib import Path
import json
import sys
from typing import Any


SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
	sys.path.insert(0, str(SOURCE_DIR))


def _yaml_module():
	try:
		import yaml
	except ImportError as error:
		raise RuntimeError(
			"PyYAML is required. Run these scripts in the project ML environment."
		) from error
	return yaml


def read_yaml(path: Path) -> Any:
	yaml = _yaml_module()
	with path.open(encoding="utf-8") as file:
		return yaml.safe_load(file)


def read_records(path: Path) -> list[dict[str, Any]]:
	if path.suffix == ".jsonl":
		with path.open(encoding="utf-8") as file:
			data = [json.loads(line) for line in file if line.strip()]
	else:
		data = read_yaml(path)
	if not isinstance(data, list):
		raise ValueError(f"Expected a record list in {path}")
	if not all(isinstance(record, dict) for record in data):
		raise ValueError(f"Every record in {path} must be a mapping")
	return data


def write_yaml(path: Path, data: Any) -> None:
	yaml = _yaml_module()

	class NoAliasSafeDumper(yaml.SafeDumper):
		def ignore_aliases(self, value: Any) -> bool:
			return True

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		yaml.dump(
			data,
			file,
			Dumper=NoAliasSafeDumper,
			allow_unicode=True,
			sort_keys=False,
			width=1_000_000,
		)


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
	if path.suffix == ".jsonl":
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8") as file:
			for record in records:
				file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
				file.write("\n")
	else:
		write_yaml(path, records)


def require_string(record: dict[str, Any], key: str, record_index: int) -> str:
	value = record.get(key)
	if not isinstance(value, str):
		raise ValueError(f"Record {record_index} has invalid {key!r}")
	return value


def finish(step: int, records: list[dict[str, Any]], output_file: Path) -> None:
	write_records(output_file, records)
	print(f"Step {step}: wrote {len(records)} records to {output_file}")
