"""
Shared settings for the whole project. Put this file in the project root and import it
everywhere (from the root:  python -m <folder>.<script>), e.g.

    from config import H, OOS_START, MODEL_DATA_DIR, FEATURE_SETS, feature_columns

Change values here, never locally in a script, so every model uses the same setup.
"""

from pathlib import Path

import pandas as pd

# ---------- Paths (relative to this file, so they work from any working directory) ----------
ROOT = Path(__file__).resolve().parents[1]  # config.py lives in Group Project\config\
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"                # {asset}_clean/<symbol>.parquet
FEATURE_DIR = DATA_DIR / "features"           # own-symbol features, from build_features.py
MODEL_DATA_DIR = DATA_DIR / "model_data"      # + market / other-symbol features; models read this
RESULTS_DIR = ROOT / "results" / "forecasts"
FIG_DIR = ROOT / "results" / "figures"

ASSET_CLASSES = ["fx", "futures", "stocks"]

# ---------- Sample and target ----------
START = pd.Timestamp("2015-01-02")   # first date kept, applied after features are built
H = 5                                # forecast horizon (rows); target = log vol over t+1..t+H
EPS = 1e-12                          # guards against log(0)

# ---------- Walk-forward ----------
OOS_START = pd.Timestamp("2020-01-01")  # first out-of-sample date, same for every symbol
TRAIN_SIZE = 1232                       # rolling window length (rows)
TEST_SIZE = 21                          # rows forecast before each refit
STEP_SIZE = 21                          # = TEST_SIZE, so test blocks don't overlap
GAP = H                                 # training rows dropped before each test block
INNER_CV_SPLITS = 5                     # TimeSeriesSplit folds for tuning (with gap=H)

# ---------- Feature definitions ----------
HAR_FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m"]
EXTRA_FEATURES = ["log_rsp_d", "log_rsn_d", "jump_share", "log_gkr"]

# Market averages: for each symbol, the average of the OTHER symbols in the same
# asset class on that date (leave-one-out), for each HAR term.
MKT_SOURCE_TERMS = HAR_FEATURES
MKT_FEATURES = ["log_rv_mkt_d", "log_rv_mkt_w", "log_rv_mkt_m"]

# Other-symbol volatility: one column per other symbol in the asset class,
# named other_<term>_<SYMBOL>, e.g. other_log_rv_d_GBPUSD.
OTHER_PREFIX = "other_"
OTHER_SYMBOL_TERMS = ["log_rv_d"]

# Days a missing value from another symbol (e.g. its holiday) may be carried forward.
# Carrying forward only uses past data, so it doesn't leak.
FFILL_LIMIT = 3

FEATURE_SETS = {
    "har": HAR_FEATURES,
    "har_ext": HAR_FEATURES + EXTRA_FEATURES,
    "har_mkt": HAR_FEATURES + EXTRA_FEATURES + MKT_FEATURES + [OTHER_PREFIX],  # prefix expands per symbol
}


def feature_columns(set_name: str, columns) -> list[str]:
    """
    Resolve a feature set to actual column names for one symbol's table.
    Other-symbol columns differ by asset class (fx has 4, stocks may have many),
    so the prefix entry is expanded against the table's columns.

        cols = feature_columns("har_mkt", df.columns)
    """
    cols = []
    for name in FEATURE_SETS[set_name]:
        if name == OTHER_PREFIX:
            cols += sorted(c for c in columns if c.startswith(OTHER_PREFIX))
        else:
            cols.append(name)
    missing = [c for c in cols if c not in columns]
    if missing:
        raise KeyError(f"Feature set '{set_name}' needs columns not in the data: {missing}")
    return cols