from pathlib import Path

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
# Path to project root (one level up from this bls package directory)
PROJECT_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = PROJECT_DIR / "instances"
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

INSTANCE_NAME = "timtab1.mps"
INSTANCE_PATH = INSTANCE_DIR / INSTANCE_NAME

# =====================================================================
# SOLVER & ALGORITHM PARAMETERS
# =====================================================================
TIME_LIMIT = 6
THREADS = 1
OUTPUT_FLAG = 1
SEED = 1
NUMERIC_FOCUS = 2
MIP_GAP = 0.0
EPS = 1e-6

# BLS Heuristic specific parameters
USE_BLS = True
VERBOSE = False
MIN_LP_POINTS = 4
BLS_MAX_CANDIDATES = 200
BLS_MAX_EXTENSION = 50

# Tabu Search parameters
USE_TABU_LOCAL_SEARCH = True
TABU_MAX_STEPS = 15
TABU_TENURE = 5
