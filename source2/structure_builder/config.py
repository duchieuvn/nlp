from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

ANNOTATIONS_FILE = DATA_DIR / "2_annotations.json"
EQUATIONS_FILE = DATA_DIR / "3_equations.json"
HTML_DIR = DATA_DIR / "html"

SOURCE2_DATA_DIR = DATA_DIR / "source2"
STRUCTURED_PAPERS_DIR = SOURCE2_DATA_DIR / "structured_papers"
