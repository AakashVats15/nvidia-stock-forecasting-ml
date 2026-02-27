import os
import joblib
import pandas as pd
from config import DATA_TARGETS, MODEL_DIR, MODELS, DROP, SPLIT
import numpy as np

def load(path):
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")

def split(df):
    n = len(df)
    t1 = int(n * SPLIT[0])
    t2 = int(n * SPLIT[1])
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]

def prepare(df):
    y = df["target_ret_1d"]
    X = df.drop(DROP, axis=1)
    return X, y

def reshape_lstm(X, y, window=20):
    X = X.values
    y = y.values
    Xs = []
    ys = []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_model(name, X, y):
    m = MODELS[name]()
    if name == "arima":
        return m(y)
    if name == "lstm":
        X, y = reshape_lstm(X, y)
        m.fit(X, y, epochs=20, batch_size=32, verbose=0)
        return m
    m.fit(X, y)
    return m

def run(inp=DATA_TARGETS, out_dir=MODEL_DIR, model_name="linear"):
    df = load(inp)
    tr, va, te = split(df)
    Xtr, ytr = prepare(tr)
    m = train_model(model_name, Xtr, ytr)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(m, os.path.join(out_dir, f"{model_name}.pkl"))

if __name__ == "__main__":
    run()