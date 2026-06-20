from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
SOURCE2_DATA_DIR = PROJECT_DIR / "data" / "source2"
STRUCTURED_PAPERS_DIR = SOURCE2_DATA_DIR / "structured_papers"
SYMBOLS_DIR = SOURCE2_DATA_DIR / "symbols"
RELATIONS_DIR = SOURCE2_DATA_DIR / "relations"

STRONG_THRESHOLD = 5.0
POTENTIAL_THRESHOLD = 2.0
VALID_GRADES = {"strong", "potential", "none"}
VALID_DESCRIPTIONS = {
	"explicit citation",
	"derived from",
	"equivalent",
	"special case",
	"shares symbols",
	"same section context",
	"",
}
