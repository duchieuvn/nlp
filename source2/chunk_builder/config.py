from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
SOURCE2_DATA_DIR = PROJECT_DIR / "data" / "source2"
STRUCTURED_PAPERS_DIR = SOURCE2_DATA_DIR / "structured_papers"
CHUNKS_DIR = SOURCE2_DATA_DIR / "chunks"
