import pandas as pd
import numpy as np

def load(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df

def compute(df):
    df["ret"] = df["Close"].pct_change()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()

def run(raw="data/raw/NVDA.csv", out="data/processed/nvda.csv"):
    df = load(raw)
    df = compute(df)
    df.to_csv(out)

if __name__ == "__main__":
    run()