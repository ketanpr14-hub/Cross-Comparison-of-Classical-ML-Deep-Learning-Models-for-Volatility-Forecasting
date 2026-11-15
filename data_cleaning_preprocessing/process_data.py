"""
Build features (X) and target (y) for every cleaned series.

Reads:  data/clean/{fx,futures,stocks}_clean/<symbol>.parquet
Writes: data/features/{fx,futures,stocks}_features/<symbol>.parquet

Each output row is "standing at the close of day t":
  - features use data from day t and earlier
  - target y = log of realized volatility over days t+1 ... t+5

Assumes rv5, rsp5, rsn5, bv5 are daily VARIANCES (not volatilities).
Check the dataset documentation and adjust if not.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---- Shared settings (move these into config.py) ----
CLEAN_DIR = Path("data/clean")
FEATURE_DIR = Path("data/features")
ASSET_CLASSES = ["fx", "futures", "stocks"]
START = pd.Timestamp("2015-01-02")
H = 5        # forecast horizon in observations
EPS = 1e-12  # guards against log(0)


def safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=0) + EPS) # Safeguard against any 0 logs, otherwise use very small value


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df = one symbol, one row per date."""
    df = df.sort_values("date").reset_index(drop=True)
    out = df[["date", "symbol"]].copy() # Pair by date and symbol (ex. AUDUSD, AAPL, etc.)

    rv = df["rv5"] # Target Variable
    
    # HAR terms: average over windows, then log (days t-k+1 ... t)
    out["log_rv_d"] = safe_log(rv) 
    out["log_rv_w"] = safe_log(rv.rolling(5).mean())
    out["log_rv_m"] = safe_log(rv.rolling(22).mean())

    # Optional extras
    out["log_rsp_d"] = safe_log(df["rsp5"])                # "good" semivariance
    out["log_rsn_d"] = safe_log(df["rsn5"])                # "bad" semivariance
    jump = (rv - df["bv5"]).clip(lower=0)
    out["jump_share"] = jump / (rv + EPS)                  # share of RV from jumps, 0 to 1
    out["log_gkr"] = safe_log(df["gkr"])                   # range-based estimator

    # Target: rv5 summed over t+1 ... t+H, moved back to row t
    fwd_sum = rv.rolling(H).sum().shift(-H)
    out["y"] = np.log(np.sqrt(fwd_sum + EPS)) # Volatility is the square root of the variance

    return out


def main() -> None:
    for asset_class in ASSET_CLASSES:
        in_dir = CLEAN_DIR / f"{asset_class}_clean"
        out_dir = FEATURE_DIR / f"{asset_class}_features"
        out_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(in_dir.glob("*.parquet")):
            df = pd.read_parquet(path)

            n_zero = int((df["rv5"] <= 0).sum())
            if n_zero:
                print(f"  warning: {path.name} has {n_zero} rows with rv5 <= 0")

            feats = build_features(df)

            # Filter AFTER building features so early rolling windows have history
            feats = feats[feats["date"] >= START]
            # Drop rows without full features (start) or without a target (last H rows)
            feats = feats.dropna().reset_index(drop=True)

            feats.to_parquet(out_dir / path.name, index=False)
            print(f"{asset_class:8s} {path.stem:12s} {len(feats):6d} rows  "
                  f"{feats['date'].min().date()} to {feats['date'].max().date()}")


if __name__ == "__main__":
    main()