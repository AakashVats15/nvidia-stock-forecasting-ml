import os
import pandas as pd
import joblib
from src.config import DATA_TARGETS, MODEL_DIR, MODELS, DROP, SPLIT
from src.train import load, split, prepare

def evaluate_model(name, model, X, y):
    if name == "arima":
        n_train = len(model.data.endog)
        n_test = len(y)
        start = n_train
        end = n_train + n_test - 1

        try:
            p = model.predict(start=start, end=end)
        except:
            return float("nan")

        p = pd.Series(p).astype(float)
        p = p.dropna()

        if len(p) == 0:
            return float("nan")

        p = p.iloc[:len(y)]
        y = y.iloc[:len(p)]

        if len(p) == 0:
            return float("nan")

        return (p - y).abs().mean()

    p = model.predict(X)
    p = pd.Series(p, index=y.index)
    return (p - y).abs().mean()

def run():
    df = load(DATA_TARGETS)
    tr, va, te = split(df)
    Xte, yte = prepare(te)

    rows = []
    for name in MODELS:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            continue
        m = joblib.load(path)
        e = evaluate_model(name, m, Xte, yte)
        rows.append([name, e])

    out = pd.DataFrame(rows, columns=["model", "mae"])
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/eval_results.csv", index=False)
    print(out)

if __name__ == "__main__":
    run()
