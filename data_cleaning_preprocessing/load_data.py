import numpy as np
import pandas as pd
from pathlib import Path

fx = pd.read_csv("../Data/realized_variance_forex_20261030/realized_variance_forex_20261030.csv")
futures = pd.read_csv("../Data/realized_variance_futures_20261030/realized_variance_futures_20261030.csv")
stocks = pd.read_csv("../Data/realized_variance_stocks_20261030/realized_variance_stocks_20261030.csv")

def preprocess_data():
    START = pd.Timestamp("2015-01-02") # Common start date for all asset classes 
    datasets = {"fx": fx, "futures": futures, "stocks": stocks} # Each asset type

    series = {}
    for asset_class, df in datasets.items():
        df['date'] = pd.to_datetime(df['date'])
        df = df[df["date"] >= START] # Make every dataframe have the same start date
        for sym, g in df.groupby("symbol"):
            series[(asset_class, sym)] = g.sort_values("date").reset_index(drop=True) # Make every symbol a separate time series, allows for more robust analysis

    CLEAN_DIR = Path("data/clean")

    for (asset_class, sym), g in series.items():
        folder = CLEAN_DIR / f"{asset_class}_clean"
        folder.mkdir(parents=True, exist_ok=True)
        g.to_parquet(folder / f"{sym}.parquet", index=False) # Save to asset-specific folders

preprocess_data()    

    