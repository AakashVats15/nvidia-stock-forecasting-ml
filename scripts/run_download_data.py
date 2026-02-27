import os
import yfinance as yf
from src.config import DATA_RAW

def run():
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    df = yf.download("NVDA", auto_adjust=False)
    df.to_csv(DATA_RAW)
    print("ok")

if __name__ == "__main__":
    run()