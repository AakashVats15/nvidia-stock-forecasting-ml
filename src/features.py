import pandas as pd
import numpy as np

def lagged(df, col, lags):
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df

def rolling(df, col, windows):
    for w in windows:
        df[f"{col}_mean{w}"] = df[col].rolling(w).mean()
        df[f"{col}_vol{w}"] = df[col].rolling(w).std()
    return df

def rsi(df, col, window=14):
    d = df[col].diff()
    u = d.clip(lower=0)
    v = -d.clip(upper=0)
    rs = u.rolling(window).mean() / v.rolling(window).mean()
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def macd(df, col):
    e12 = df[col].ewm(span=12, adjust=False).mean()
    e26 = df[col].ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df

def stoch(df, high="High", low="Low", close="Close", k=14, d=3):
    ll = df[low].rolling(k).min()
    hh = df[high].rolling(k).max()
    df["stoch_k"] = 100 * (df[close] - ll) / (hh - ll)
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df

def build_features(df):
    df = lagged(df, "ret", [1,5,10,20])
    df = rolling(df, "ret", [5,10,20])
    df = rsi(df, "Close")
    df = macd(df, "Close")
    df = stoch(df)
    return df.dropna()

def run(inp="data/processed/nvda.csv", out="data/processed/nvda_features.csv"):
    df = pd.read_csv(inp, parse_dates=["Date"], index_col="Date")
    df = build(df)
    df.to_csv(out)

if __name__ == "__main__":
    run()