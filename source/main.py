from pathlib import Path
import importlib.util


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_START_INDEX = 1
DEFAULT_DOWNLOAD_PAPER_COUNT = 100


def load_step(filename: str):
	module_name = filename.removesuffix(".py")
	module_path = SCRIPT_DIR / filename
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def run_step(step_number: int, description: str, callback) -> None:
	print(f"\n[{step_number}/4] {description}")
	callback()


def main(
	download_start_index: int = DEFAULT_DOWNLOAD_START_INDEX,
	download_paper_count: int | None = DEFAULT_DOWNLOAD_PAPER_COUNT,
) -> None:
	download = load_step("0_download.py")
	extract_enumerated_math = load_step("1_extract_enumerated_math.py")
	extract_anno = load_step("2_extract_anno.py")
	extract_equations = load_step("3_extract_equations.py")

	run_step(
		1,
		"Download HTML papers",
		lambda: download.main(download_start_index, download_paper_count),
	)
	run_step(
		2,
		"Extract enumerated math tbody blocks",
		extract_enumerated_math.main,
	)
	run_step(
		3,
		"Extract annotation tags",
		extract_anno.main,
	)
	run_step(
		4,
		"Build final equation dataset",
		extract_equations.main,
	)

	print("\nPipeline complete.")


if __name__ == "__main__":
	main()
