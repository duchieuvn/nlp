import importlib
import sys
from config import (
	DEFAULT_DOWNLOAD_PAPER_COUNT,
	DEFAULT_DOWNLOAD_START_INDEX,
	SOURCE_DIR,
)
from step0_download import download_main
from step1_extract_enumerated_math import extract_tbody
from step2_extract_anno import extract_anno_main
from step3_extract_equations import extract_eqn_main

if str(SOURCE_DIR) not in sys.path:
	sys.path.insert(0, str(SOURCE_DIR))

def main(
	download_start_index: int = DEFAULT_DOWNLOAD_START_INDEX,
	download_paper_count: int | None = DEFAULT_DOWNLOAD_PAPER_COUNT,
) -> None:
	print("\n[1/4] Download HTML papers")
	download_main(download_start_index, download_paper_count)

	print("\n[2/4] Extract enumerated math tbody blocks")
	extract_tbody()

	print("\n[3/4] Extract annotation tags")
	extract_anno_main()

	print("\n[4/4] Build final equation dataset")
	extract_eqn_main()  

	print("\nPipeline complete.")


if __name__ == "__main__":
	main()
