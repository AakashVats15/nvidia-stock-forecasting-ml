import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

def ts():
    return datetime.now().strftime("%d%m%y_%H_%M")

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

def preds(model, X):
    return model.predict(X)

def savefig(name):
    plt.savefig(f"figures/{ts()}_{name}.png")
    plt.close()

def actual_vs_pred(y, p):
    plt.figure(figsize=(12,4))
    plt.plot(y.index, y, color="blue", label="actual return")
    plt.plot(y.index, p, color="orange", label="predicted return")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    savefig("actual_vs_pred_test")

def residuals(y, p):
    r = y - p
    plt.figure(figsize=(6,4))
    plt.scatter(p, r, s=5, color="purple")
    plt.axhline(0, color="black")
    plt.xlabel("Predicted Return")
    plt.ylabel("Residual")
    plt.legend(["zero line", "residuals"])
    savefig("residuals_test")

def residual_hist(y, p):
    r = y - p
    plt.figure(figsize=(6,4))
    plt.hist(r, bins=50, color="gray")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.legend(["residual distribution"])
    savefig("residual_hist_test")

def rolling_rmse(y, p):
    r = (y - p)**2
    rm = np.sqrt(r.rolling(60).mean())
    plt.figure(figsize=(12,4))
    plt.plot(rm.index, rm, color="red", label="rolling RMSE (60d)")
    plt.xlabel("Date")
    plt.ylabel("RMSE")
    plt.legend()
    savefig("rolling_rmse_test")

def processed_vs_raw(raw, proc):
    plt.figure(figsize=(12,4))
    plt.plot(raw.index, raw["Close"], color="black", label="raw close")
    plt.plot(proc.index, proc["Close"], color="green", label="processed close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    savefig("processed_vs_raw")

def combined_price_plot(raw, proc, y, p):
    idx = y.index
    r = raw.loc[idx]["Close"]
    pr = proc.loc[idx]["Close"]
    pc = pr.shift(1) * (1 + p)
    e = np.abs(pc - pr)
    eb = e.rolling(20).mean()
    up = pc + eb
    lo = pc - eb
    m = (e / pr * 100).mean()
    plt.figure(figsize=(12,4))
    plt.plot(idx, r, color="black", label="raw close")
    plt.plot(idx, pr, color="green", label="processed close")
    plt.plot(idx, pc, color="red", label="predicted close")
    plt.fill_between(idx, lo, up, color="yellow", alpha=0.2, label="error band")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.text(idx[-1], max(r.max(), pr.max(), pc.max()), f"Error: {m:.2f}%", 
             ha="right", va="top", fontsize=10, color="blue")
    savefig("raw_processed_predicted")



def run(data="data/processed/nvda_targets.csv", model="src/models/linear_regression.pkl"):
    os.makedirs("figures", exist_ok=True)
    df = load(data)
    raw = load("data/raw/NVDA.csv")
    proc = load("data/processed/nvda.csv")
    tr, va, te = split(df)
    Xte, yte = prepare(te)
    m = joblib.load(model)
    pte = preds(m, Xte)
    actual_vs_pred(yte, pte)
    residuals(yte, pte)
    residual_hist(yte, pte)
    rolling_rmse(yte, pte)
    processed_vs_raw(raw, proc)
    combined_price_plot(raw, proc, yte, pte)

if __name__ == "__main__":
    run()