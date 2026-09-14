from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TOP10_NUMERIC_FEATURES = [
    "vitals_count",
    "heartrate",
    "sbp",
    "dbp",
    "pain",
    "resprate_max",
    "o2sat",
    "o2sat_min",
    "heartrate_max",
    "pain_min",
]

ALL_NUMERIC_CANDIDATES = [
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain",
    "temperature_first", "temperature_mean", "temperature_min", "temperature_max",
    "heartrate_first", "heartrate_mean", "heartrate_min", "heartrate_max",
    "resprate_first", "resprate_mean", "resprate_min", "resprate_max",
    "o2sat_first", "o2sat_mean", "o2sat_min", "o2sat_max",
    "sbp_first", "sbp_mean", "sbp_min", "sbp_max",
    "dbp_first", "dbp_mean", "dbp_min", "dbp_max",
    "pain_first", "pain_mean", "pain_min", "pain_max",
    "vitals_count",
]

CLIP_RANGES = {
    "temperature": (30, 43), "temperature_first": (30, 43),
    "temperature_mean": (30, 43), "temperature_min": (28, 45), "temperature_max": (30, 46),
    "heartrate": (20, 250), "heartrate_first": (20, 250), "heartrate_mean": (20, 250),
    "heartrate_min": (10, 250), "heartrate_max": (20, 280),
    "resprate": (4, 80), "resprate_first": (4, 80), "resprate_mean": (4, 80),
    "resprate_min": (2, 80), "resprate_max": (4, 120),
    "o2sat": (50, 100), "o2sat_first": (50, 100), "o2sat_mean": (50, 100),
    "o2sat_min": (30, 100), "o2sat_max": (50, 100),
    "sbp": (40, 260), "sbp_first": (40, 260), "sbp_mean": (40, 260),
    "sbp_min": (30, 260), "sbp_max": (40, 300),
    "dbp": (20, 200), "dbp_first": (20, 200), "dbp_mean": (20, 200),
    "dbp_min": (10, 200), "dbp_max": (20, 220),
    "pain": (0, 10), "pain_first": (0, 10), "pain_mean": (0, 10),
    "pain_min": (0, 10), "pain_max": (0, 10),
    "vitals_count": (0, 500),
}
