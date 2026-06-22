from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_DIR / "data" / "source2"

CHUNKS_DIR = DATA_DIR / "chunks"
SYMBOLS_DIR = DATA_DIR / "symbols"
STRUCTURED_PAPERS_DIR = DATA_DIR / "structured_papers"
SYMBOL_MEANINGS_DIR = DATA_DIR / "symbol_meanings2"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoint"
REVIEW_SAMPLE_FILE = PROJECT_DIR / "analysis" / "symbol_meaning2_review.json"

TOP_K = 12
MINIMUM_DEFINITION_SCORE = 6.0
MAX_NEURAL_CANDIDATES = 48
MAX_TOKENS = 256
MODEL_BATCH_SIZE = 16
TARGET_ACCEPTED_PRECISION = 0.90
MINIMUM_REVIEWED_ACCEPTS = 10
DEFAULT_RELATION_THRESHOLD = 0.9980401396751404
DEFAULT_RELATION_MARGIN = 0.10

# Set to True after inference_config.json is calibrated on human-reviewed data.
REQUIRE_REVIEWED_CALIBRATION = True
DEVICE = "auto"

DEFINITION_CUES = (
	"denotes", "represents", "is defined as", "is the", "are the", "where",
	"let", "called", "refers to", "stands for",
)
