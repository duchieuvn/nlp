from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
EQUATIONS_FILE = DATA_DIR / "3_equations.json"
SOURCE2_DATA_DIR = DATA_DIR / "source2"
CHUNKS_DIR = SOURCE2_DATA_DIR / "chunks"
SYMBOLS_DIR = SOURCE2_DATA_DIR / "symbols"
