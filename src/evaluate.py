import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(path):
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")

def split(df):
    n = len(df)
    t1 = int(n * 0.7)
    t2 = int(n * 0.85)
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]

def prepare(df):
    y = df["target_ret_1d"]
    X = df.drop(["target_ret_1d", "target_sumret_5d", "target_dir"], axis=1)
    return X, y

def metrics(y, p):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    r2 = r2_score(y, p)
    mape = np.mean(np.abs((y - p) / y)) * 100
    return mae, rmse, r2, mape

def run(data="data/processed/nvda_targets.csv", model="src/models/linear_regression.pkl"):
    df = load_data(data)
    tr, va, te = split(df)
    Xtr, ytr = prepare(tr)
    Xva, yva = prepare(va)
    Xte, yte = prepare(te)
    m = joblib.load(model)
    ptr = m.predict(Xtr)
    pva = m.predict(Xva)
    pte = m.predict(Xte)
    print("train", metrics(ytr, ptr))
    print("val", metrics(yva, pva))
    print("test", metrics(yte, pte))

if __name__ == "__main__":
    run()