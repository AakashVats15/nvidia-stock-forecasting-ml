import os
import pandas as pd
from src.config import DATA_RAW, DATA_PROC, DATA_FEATURES, DATA_TARGETS
from src.data_pipeline import load, compute
from src.features import build_features
from src.targets import build_targets

def run():
    df = load(DATA_RAW)
    df = compute(df)
    os.makedirs(os.path.dirname(DATA_PROC), exist_ok=True)
    df.to_csv(DATA_PROC)

    f = build_features(df.copy())
    f.to_csv(DATA_FEATURES)

    t = build_targets(df.copy())
    t.to_csv(DATA_TARGETS)

    print("ok")

if __name__ == "__main__":
    run()