import pandas as pd

def next_return(df, col="ret", horizon=1):
    df[f"target_ret_{horizon}d"] = df[col].shift(-horizon)
    return df

def next_sum_return(df, col="ret", horizon=5):
    df[f"target_sumret_{horizon}d"] = df[col].rolling(horizon).sum().shift(-horizon+1)
    return df

def direction(df, col="ret"):
    df["target_dir"] = (df[col].shift(-1) > 0).astype(int)
    return df

def build_targets(df):
    df = next_return(df)
    df = next_sum_return(df)
    df = direction(df)
    return df.dropna()

def run(inp="data/processed/nvda_features.csv", out="data/processed/nvda_targets.csv"):
    df = pd.read_csv(inp, parse_dates=["Date"], index_col="Date")
    df = build_targets(df)
    df.to_csv(out)

if __name__ == "__main__":
    run()
