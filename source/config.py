from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SOURCE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
INPUT_PREPROCESS_DATA_DIR = DATA_DIR / "input_preprocess"
LIST_FILE = DATA_DIR / "paper_list_46.txt"
HTML_DIR = DATA_DIR / "html"
ERROR_LOG = DATA_DIR / "papers_error.log"

HTML_MATH_FILE = DATA_DIR / "1_html_math.json"
ANNOTATIONS_FILE = DATA_DIR / "2_annotations.json"
EQUATIONS_FILE = DATA_DIR / "3_equations.json"
EQUATION_WINDOWS_FILE = DATA_DIR / "equation_windows.yaml"
PREPROCESS_STEP_1_FILE = INPUT_PREPROCESS_DATA_DIR / "1_loaded_windows.yaml"
PREPROCESS_STEP_2_FILE = INPUT_PREPROCESS_DATA_DIR / "2_equation_boundaries.yaml"
PREPROCESS_STEP_3_FILE = INPUT_PREPROCESS_DATA_DIR / "3_candidates.yaml"
PREPROCESS_STEP_4_FILE = INPUT_PREPROCESS_DATA_DIR / "4_contexts.jsonl"
PREPROCESS_STEP_5_FILE = INPUT_PREPROCESS_DATA_DIR / "5_model_texts.jsonl"
PREPROCESS_STEP_6_FILE = INPUT_PREPROCESS_DATA_DIR / "6_tokenized.jsonl"
PREPROCESS_STEP_7_FILE = INPUT_PREPROCESS_DATA_DIR / "7_model_inputs.jsonl"
EQUATION_MEANINGS_FILE = DATA_DIR / "4_equation_meanings.json"
EQUATION_MEANINGS_B_FILE = DATA_DIR / "4b_equation_meanings.json"

DEFAULT_DOWNLOAD_START_INDEX = 1
DEFAULT_DOWNLOAD_PAPER_COUNT = 100
BASE_URL = "https://arxiv.org/html/{id}"
