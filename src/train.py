import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def load(path):
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

def train(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m

def run(inp="data/processed/nvda_targets.csv", out="src/models/linear_regression.pkl"):
    df = load(inp)
    tr, va, te = split(df)
    Xtr, ytr = prepare(tr)
    m = train(Xtr, ytr)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(m, out)

if __name__ == "__main__":
    run()
