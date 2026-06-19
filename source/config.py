from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SOURCE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
LIST_FILE = DATA_DIR / "paper_list_46.txt"
HTML_DIR = DATA_DIR / "html"
ERROR_LOG = DATA_DIR / "papers_error.log"

HTML_MATH_FILE = DATA_DIR / "1_html_math.json"
ANNOTATIONS_FILE = DATA_DIR / "2_annotations.json"
EQUATIONS_FILE = DATA_DIR / "3_equations.json"
EQUATION_MEANINGS_FILE = DATA_DIR / "4_equation_meanings.json"

DEFAULT_DOWNLOAD_START_INDEX = 1
DEFAULT_DOWNLOAD_PAPER_COUNT = 100
BASE_URL = "https://arxiv.org/html/{id}"
