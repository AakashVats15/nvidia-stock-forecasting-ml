import os
from src.models.linear_models import linear, ridge, lasso, elastic
from src.models.tree_models import rf, gbr
from src.models.lstm_model import lstm
from src.models.arima_models import arima

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
    "linear": linear,
    "ridge": ridge,
    "lasso": lasso,
    "elastic": elastic,
    "rf": rf,
    "gbr": gbr,
    "lstm": lstm,
    "arima": arima
}