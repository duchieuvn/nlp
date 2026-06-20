from pathlib import Path


SOURCE2_DIR = Path(__file__).parent.parent
PROJECT_DIR = SOURCE2_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
EQUATIONS_FILE = DATA_DIR / "3_equations.json"
RETRIEVAL_RESULTS_DIR = DATA_DIR / "source2" / "retrieval_results"
EQUATION_MEANINGS_DIR = DATA_DIR / "source2" / "equation_meanings"

MINIMUM_CANDIDATE_SCORE = 6.0

MATHBERT_MODEL_NAME = "witiko/mathberta"
MATHBERT_MAX_TOKENS = 256
MATHBERT_BATCH_SIZE = 16
MATHBERT_SCORE_WEIGHT = 2.0
