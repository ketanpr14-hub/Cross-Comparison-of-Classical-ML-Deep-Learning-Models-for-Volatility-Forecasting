"""
k-nearest-neighbors forecasts through the rolling walk-forward loop, for every feature set.

Two versions, so the extrapolation effect can be compared:
  knn_level:  neighbors' average y (log vol over the next 5 days)
  knn_change: current level + neighbors' average change in log vol.
              Can forecast above anything seen in training (e.g. March 2020).

Reads:  data/model_data/{fx,futures,stocks}_model_data/<symbol>.parquet
Writes: results/forecasts/<asset_class>_<symbol>_knn.parquet
        columns: date, asset_class, symbol, model, feature_set, fold, k, weights, actual, forecast

In each fold, inside the training rows only:
  1. Time-series CV picks k and the weighting
  2. The scaler and model are refit on all training rows with those settings
  3. The test rows are forecast
Settings and feature sets come from config.py.
Run from the project root:  python -m <folder>.knn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (ASSET_CLASSES, FEATURE_SETS, H, INNER_CV_SPLITS, MODEL_DATA_DIR,
                    RESULTS_DIR, feature_columns)
from data_cleaning_preprocessing.walk_forward import walk_forward_generator

# Feature sets k-NN runs on. Distances lose meaning as features are added, so if
# har_mkt gets wide (many stock symbols), consider a smaller set here.
KNN_FEATURE_SETS = list(FEATURE_SETS)

INNER_CV = TimeSeriesSplit(n_splits=INNER_CV_SPLITS, gap=H)
PARAM_GRID = {
    "model__n_neighbors": [5, 10, 20, 50, 100],
    "model__weights": ["uniform", "distance"],
}


# ---------- Helpers ----------

def naive_level(df):
    """Last week's log vol, in the same units as y. log_rv_w = log(mean rv5 over t-4..t),
    so log(sqrt(5-day sum)) = 0.5 * (log(5) + log_rv_w)."""
    return 0.5 * (np.log(H) + df["log_rv_w"].to_numpy())


def fit_knn(X_train, y_train, X_test):
    pipe = Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())])
    search = GridSearchCV(pipe, PARAM_GRID, cv=INNER_CV,
                          scoring="neg_mean_squared_error", n_jobs=-1)
    search.fit(X_train, y_train)
    params = {"k": search.best_params_["model__n_neighbors"],
              "weights": search.best_params_["model__weights"]}
    return search.predict(X_test), params


# ---------- Models: each takes (train_df, test_df, cols), returns (forecasts, chosen params) ----------

def knn_level_model(train_df, test_df, cols):
    return fit_knn(train_df[cols], train_df["y"], test_df[cols])


def knn_change_model(train_df, test_df, cols):
    change_train = train_df["y"].to_numpy() - naive_level(train_df)
    pred_change, params = fit_knn(train_df[cols], change_train, test_df[cols])
    return naive_level(test_df) + pred_change, params


MODELS = {"knn_level": knn_level_model, "knn_change": knn_change_model}


# ---------- Walk-forward runner ----------

def run_walk_forward(df, model_fn, model_name, set_name, cols):
    pieces = []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_generator(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        forecasts, params = model_fn(train_df, test_df, cols)
        pieces.append(pd.DataFrame({
            "date": test_df["date"].to_numpy(),
            "model": model_name,
            "feature_set": set_name,
            "fold": fold,
            "k": params["k"],
            "weights": params["weights"],
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


# ---------- Main ----------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for asset_class in ASSET_CLASSES:
        in_dir = MODEL_DATA_DIR / f"{asset_class}_model_data"

        for path in sorted(in_dir.glob("*.parquet")):
            df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)

            pieces = []
            for set_name in KNN_FEATURE_SETS:
                cols = feature_columns(set_name, df.columns)
                for model_name, fn in MODELS.items():
                    pieces.append(run_walk_forward(df, fn, model_name, set_name, cols))
            forecasts = pd.concat(pieces, ignore_index=True)
            forecasts.insert(1, "asset_class", asset_class)
            forecasts.insert(2, "symbol", path.stem)
            forecasts.to_parquet(RESULTS_DIR / f"{asset_class}_{path.stem}_knn.parquet", index=False)

            # Summary vs naive on the same out-of-sample dates
            oos = df[df["date"].isin(forecasts["date"])]
            naive_mse = mse_log(oos["y"], naive_level(oos))
            naive_qlike = qlike(oos["y"], naive_level(oos))

            summary = forecasts.groupby(["model", "feature_set"], sort=False).apply(
                lambda g: pd.Series({
                    "mse": mse_log(g["actual"], g["forecast"]),
                    "qlike": qlike(g["actual"], g["forecast"]),
                    "median_k": g.groupby("fold")["k"].first().median(),
                }),
                include_groups=False,
            )
            summary["mse_ratio"] = summary["mse"] / naive_mse
            summary["qlike_ratio"] = summary["qlike"] / naive_qlike
            print(f"\n{asset_class} {path.stem}")
            print(summary.round(4))


if __name__ == "__main__":
    main()