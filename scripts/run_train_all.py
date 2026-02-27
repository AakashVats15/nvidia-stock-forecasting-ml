import os
import joblib
import pandas as pd
from src.config import DATA_TARGETS, MODEL_DIR, MODELS, DROP, SPLIT
from src.train import load, split, prepare, train_model

def run():
    df = load(DATA_TARGETS)
    tr, va, te = split(df)
    Xtr, ytr = prepare(tr)

    rows = []
    for name in MODELS:
        try:
            m = train_model(name, Xtr, ytr)
            path = os.path.join(MODEL_DIR, f"{name}.pkl")
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(m, path)
            rows.append([name, "ok"])
        except Exception as e:
            rows.append([name, "fail"])

    out = pd.DataFrame(rows, columns=["model", "status"])
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/train_results.csv", index=False)
    print(out)

if __name__ == "__main__":
    run()