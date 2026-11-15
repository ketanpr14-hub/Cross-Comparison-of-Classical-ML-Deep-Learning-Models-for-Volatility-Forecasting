"""
Naive and OLS (HAR) forecasts through the rolling walk-forward loop.

  naive: next week's vol = last week's vol. Nothing is fitted; feature_set = "none".
  har:   OLS of y on a feature set. On the "har" set this is the classic HAR model;
         on har_ext / har_mkt it is HAR-X, the linear benchmark with the same
         information ridge, lasso and the ML models get.

Reads:  data/model_data/{fx,futures,stocks}_model_data/<symbol>.parquet
Writes: results/forecasts/<asset_class>_<symbol>_baseline.parquet
        columns: date, asset_class, symbol, model, feature_set, fold, actual, forecast

Settings and feature sets come from config.py.
Run from the project root:  python -m <folder>.naive_baseline_and_ols
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import ASSET_CLASSES, FEATURE_SETS, H, MODEL_DATA_DIR, RESULTS_DIR, feature_columns
from data_cleaning_preprocessing.walk_forward import walk_forward_generator
from sklearn.metrics import mean_squared_error

# Feature sets OLS runs on. Use ["har"] for the classic HAR model only.
OLS_FEATURE_SETS = list(FEATURE_SETS)


# ---------- Models: each takes (train_df, test_df, cols), returns forecasts for test_df ----------

def naive_model(train_df, test_df, cols=None):
    """log_rv_w is log(mean rv5 over t-4..t), so log(sqrt(5-day sum)) = 0.5 * (log(5) + log_rv_w).
    Nothing is fitted and no feature set is used."""
    return 0.5 * (np.log(H) + test_df["log_rv_w"].to_numpy())


def ols_model(train_df, test_df, cols):
    """OLS of y on the given columns, fitted on the training rows only."""
    X_train = sm.add_constant(train_df[cols], has_constant="add")
    X_test = sm.add_constant(test_df[cols], has_constant="add")
    results = sm.OLS(train_df["y"], X_train).fit()
    return results.predict(X_test).to_numpy()


# ---------- Walk-forward runner ----------

def run_walk_forward(df, model_fn, model_name, set_name, cols):
    pieces = []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_generator(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        pieces.append(pd.DataFrame({
            "date": test_df["date"].to_numpy(),
            "model": model_name,
            "feature_set": set_name,
            "fold": fold,
            "actual": test_df["y"].to_numpy(),
            "forecast": model_fn(train_df, test_df, cols),
        }))
    return pd.concat(pieces, ignore_index=True)


# ---------- Losses ----------

def mse_log(actual, forecast):
    return np.mean((actual - forecast) ** 2)


def qlike(actual, forecast):
    """QLIKE on variances. Inputs are log vols, so variance = exp(2 * log vol)."""
    ratio = np.exp(2 * actual) / np.exp(2 * forecast)
    return np.mean(ratio - np.log(ratio) - 1)


# ---------- Main ----------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for asset_class in ASSET_CLASSES:
        in_dir = MODEL_DATA_DIR / f"{asset_class}_model_data"

        for path in sorted(in_dir.glob("*.parquet")):
            df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)

            pieces = [run_walk_forward(df, naive_model, "naive", "none", None)]
            for set_name in OLS_FEATURE_SETS:
                cols = feature_columns(set_name, df.columns)
                pieces.append(run_walk_forward(df, ols_model, "har", set_name, cols))

            forecasts = pd.concat(pieces, ignore_index=True)
            forecasts.insert(1, "asset_class", asset_class)
            forecasts.insert(2, "symbol", path.stem)
            forecasts.to_parquet(RESULTS_DIR / f"{asset_class}_{path.stem}_baseline.parquet",
                                 index=False)

            # Quick summary: losses and ratio to naive
            summary = forecasts.groupby(["model", "feature_set"], sort=False).apply(
                lambda g: pd.Series({"mse": mse_log(g["actual"], g["forecast"]),
                                     "qlike": qlike(g["actual"], g["forecast"])}),
                include_groups=False,
            )
            naive = summary.loc[("naive", "none")]
            summary["mse_ratio"] = summary["mse"] / naive["mse"]
            summary["qlike_ratio"] = summary["qlike"] / naive["qlike"]
            print(f"\n{asset_class} {path.stem}")
            print(summary.round(4))


if __name__ == "__main__":
    main()