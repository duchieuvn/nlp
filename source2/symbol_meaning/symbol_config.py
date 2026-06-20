from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
SOURCE2_DATA_DIR = PROJECT_DIR / "data" / "source2"
CHUNKS_DIR = SOURCE2_DATA_DIR / "chunks"
SYMBOLS_DIR = SOURCE2_DATA_DIR / "symbols"
SYMBOL_MEANINGS_DIR = SOURCE2_DATA_DIR / "symbol_meanings"

TOP_K = 12
MINIMUM_DEFINITION_SCORE = 6.0
DEFINITION_CUES = (
	"denotes",
	"represents",
	"is defined as",
	"is the",
	"are the",
	"where",
)
