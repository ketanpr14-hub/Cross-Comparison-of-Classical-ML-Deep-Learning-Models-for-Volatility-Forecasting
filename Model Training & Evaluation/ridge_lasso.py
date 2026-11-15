"""
Ridge and lasso forecasts through the rolling walk-forward loop, for every feature set.

Reads:  data/model_data/{fx,futures,stocks}_model_data/<symbol>.parquet
Writes: results/forecasts/<asset_class>_<symbol>_ridge_lasso.parquet
        columns: date, asset_class, symbol, model, feature_set, fold, alpha, actual, forecast

In each fold, inside the training rows only:
  1. Time-series CV picks the penalty alpha
  2. The scaler and model are refit on all training rows with that alpha
  3. The test rows are forecast
Settings and feature sets come from config.py.
Run from the project root:  python -m <folder>.ridge_lasso
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (ASSET_CLASSES, FEATURE_SETS, H, INNER_CV_SPLITS, MODEL_DATA_DIR,
                    RESULTS_DIR, feature_columns)
from data_cleaning_preprocessing.walk_forward import walk_forward_generator

# Inner CV: time-ordered folds, with a gap so CV training targets don't overlap validation rows
INNER_CV = TimeSeriesSplit(n_splits=INNER_CV_SPLITS, gap=H)

# Log-spaced penalty grids. Lasso needs smaller values: on standardized features
# with a target this scale, alpha around 0.1 already sets most coefficients to zero.
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
LASSO_ALPHAS = np.logspace(-5, -1, 13)


# ---------- Models: each takes (train_df, test_df, cols), returns (forecasts, chosen alpha) ----------

def fit_penalized(estimator, alphas, train_df, test_df, cols):
    pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    search = GridSearchCV(
        pipe,
        param_grid={"model__alpha": alphas},
        cv=INNER_CV,
        scoring="neg_mean_squared_error",
        n_jobs=-1,  # use all CPU cores
    )
    # refit=True (the default) refits the pipeline on all training rows with the best alpha
    search.fit(train_df[cols], train_df["y"])
    forecasts = search.predict(test_df[cols])
    return forecasts, search.best_params_["model__alpha"]


def ridge_model(train_df, test_df, cols):
    return fit_penalized(Ridge(), RIDGE_ALPHAS, train_df, test_df, cols)


def lasso_model(train_df, test_df, cols):
    return fit_penalized(Lasso(max_iter=10_000), LASSO_ALPHAS, train_df, test_df, cols)


MODELS = {"ridge": ridge_model, "lasso": lasso_model}


# ---------- Walk-forward runner ----------

def run_walk_forward(df, model_fn, model_name, set_name, cols):
    pieces = []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_generator(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        forecasts, alpha = model_fn(train_df, test_df, cols)
        pieces.append(pd.DataFrame({
            "date": test_df["date"].to_numpy(),
            "model": model_name,
            "feature_set": set_name,
            "fold": fold,
            "alpha": alpha,
            "actual": test_df["y"].to_numpy(),
            "forecast": forecasts,
        }))
    return pd.concat(pieces, ignore_index=True)


# ---------- Losses ----------

def mse_log(actual, forecast):
    return np.mean((actual - forecast) ** 2)


def qlike(actual, forecast):
    """QLIKE on variances. Inputs are log vols, so variance = exp(2 * log vol)."""
    ratio = np.exp(2 * actual) / np.exp(2 * forecast)
    return np.mean(ratio - np.log(ratio) - 1)


def naive_forecast(df):
    """Same naive forecast as baseline_models.py, used here only for the printed ratios."""
    return 0.5 * (np.log(H) + df["log_rv_w"].to_numpy())


# ---------- Main ----------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for asset_class in ASSET_CLASSES:
        in_dir = MODEL_DATA_DIR / f"{asset_class}_model_data"

        for path in sorted(in_dir.glob("*.parquet")):
            df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)

            pieces = []
            for set_name in FEATURE_SETS:
                cols = feature_columns(set_name, df.columns)
                for model_name, fn in MODELS.items():
                    pieces.append(run_walk_forward(df, fn, model_name, set_name, cols))
            forecasts = pd.concat(pieces, ignore_index=True)
            forecasts.insert(1, "asset_class", asset_class)
            forecasts.insert(2, "symbol", path.stem)
            # Own file, so baseline_models.py's naive/HAR forecasts aren't overwritten
            forecasts.to_parquet(RESULTS_DIR / f"{asset_class}_{path.stem}_ridge_lasso.parquet",
                                 index=False)

            # Summary vs naive on the same out-of-sample dates
            oos = df[df["date"].isin(forecasts["date"])]
            naive_mse = mse_log(oos["y"], naive_forecast(oos))
            naive_qlike = qlike(oos["y"], naive_forecast(oos))

            summary = forecasts.groupby(["model", "feature_set"], sort=False).apply(
                lambda g: pd.Series({
                    "mse": mse_log(g["actual"], g["forecast"]),
                    "qlike": qlike(g["actual"], g["forecast"]),
                    "median_alpha": g.groupby("fold")["alpha"].first().median(),
                }),
                include_groups=False,
            )
            summary["mse_ratio"] = summary["mse"] / naive_mse
            summary["qlike_ratio"] = summary["qlike"] / naive_qlike
            print(f"\n{asset_class} {path.stem}")
            print(summary.round(4))


if __name__ == "__main__":
    main()