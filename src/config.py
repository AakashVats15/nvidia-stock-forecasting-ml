import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_RAW = os.path.join(ROOT, "data/raw/NVDA.csv")
DATA_PROC = os.path.join(ROOT, "data/processed/nvda.csv")
DATA_FEATURES = os.path.join(ROOT, "data/processed/nvda_features.csv")
DATA_TARGETS = os.path.join(ROOT, "data/processed/nvda_targets.csv")

MODEL_DIR = os.path.join(ROOT, "src/models")
FIG_DIR = os.path.join(ROOT, "figures")

SPLIT = (0.7, 0.85)

TARGET = "target_ret_1d"

DROP = ["target_ret_1d", "target_sumret_5d", "target_dir"]

MODELS = {
    "linear": LinearRegression,
    "rf": lambda: RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
}