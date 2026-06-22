from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
SOURCE2_DATA_DIR = PROJECT_DIR / "data" / "source2"

SYMBOLS_DIR = SOURCE2_DATA_DIR / "symbols"
SYMBOL_MEANINGS_DIR = SOURCE2_DATA_DIR / "symbol_meanings"
REJECTED_EVIDENCE_FILE = PROJECT_DIR / "analysis" / "rejected_symbol_meanings.json"

OUTPUT_DIR = SOURCE2_DATA_DIR / "symbol_meaning_finetune"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"
DATASET_SUMMARY_FILE = OUTPUT_DIR / "dataset_summary.json"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
REPORT_FILE = OUTPUT_DIR / "performance_report.md"
INFERENCE_CONFIG_FILE = CHECKPOINT_DIR / "inference_config.json"

MODEL_NAME = "witiko/mathberta"
# Use "auto", "cpu", or "cuda". Auto probes CUDA and falls back to CPU when
# a driver is visible but no device is allocated to the current process.
DEVICE = "auto"
MAX_TOKENS = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
MAX_REJECTED_CANDIDATES_PER_SYMBOL = 6

TARGET_ACCEPTED_PRECISION = 0.90
MINIMUM_CALIBRATION_ACCEPTS = 5
DEFAULT_RELATION_MARGIN = 0.10

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10

LABELS = (
	"DEFINES_COMPLETE_SYMBOL",
	"DEFINES_BASE",
	"QUALIFIES_SUBSCRIPT",
	"QUALIFIES_SUPERSCRIPT",
	"RELATED_NOT_DEFINITION",
	"NO_RELATION",
)
