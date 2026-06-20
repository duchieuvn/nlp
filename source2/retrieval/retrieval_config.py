from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
CHUNKS_DIR = PROJECT_DIR / "data" / "source2" / "chunks"
SYMBOLS_DIR = PROJECT_DIR / "data" / "source2" / "symbols"
RETRIEVAL_RESULTS_DIR = PROJECT_DIR / "data" / "source2" / "retrieval_results"

DEFAULT_CHUNK_TYPES = (
	"sentence",
	"paragraph",
	"equation_neighborhood",
)
