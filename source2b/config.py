from pathlib import Path

SOURCE2B_DIR = Path(__file__).parent
PROJECT_DIR = SOURCE2B_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

ANNOTATIONS_FILE = DATA_DIR / "2_annotations.json"
EQUATIONS_FILE = DATA_DIR / "3_equations.json"
HTML_DIR = DATA_DIR / "html"

OUTPUT_BASE = DATA_DIR / "source2"
DOCUMENTS_DIR = OUTPUT_BASE / "structured_papers"
CHUNKS_DIR = OUTPUT_BASE / "chunks"
RETRIEVAL_RESULTS_DIR = OUTPUT_BASE / "retrieval_results"
EQUATION_MEANINGS_DIR = OUTPUT_BASE / "equation_meanings"
SYMBOLS_DIR = OUTPUT_BASE / "symbols"
SYMBOL_MEANINGS_DIR = OUTPUT_BASE / "symbol_meanings"
POSTPROCESSED_MEANINGS_DIR = DATA_DIR / "postprocessing" / "equation_meanings"

BUILD_REPORT = OUTPUT_BASE / "build_report.json"

RETRIEVAL_METHOD = "bm25"
RETRIEVAL_TOP_K = 10
SYMBOL_MEANING_TOP_K = 12
