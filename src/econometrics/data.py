"""
Data cleaning and feature engineering.

    python -m src.econometrics.data --dataset futures      # prints what the cleaning does

`build_panel` returns one row per asset-day, indexed like the other branches: the row of
day t holds what is known at the close of t, and `target` is the log realized variance of
the next session. Stocks never mix: every lag is computed inside its own symbol.
"""
import argparse

import numpy as np
import pandas as pd

from .config import BAD_DAYS, DATASETS, JUMP_FILTER, MEASURES, RunConfig


# ------------------------------------------------------------------ 1. load
def load_raw(dataset, path=None):
    path = path or DATASETS[dataset]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    if df.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{path} contains duplicate symbol/date rows.")
    return df


# ------------------------------------------------------------------ 2. clean
def clean(df, dataset):
    """Blank the asset-days we do not trust. Returns the frame and the list of masked days."""
    df = df.copy()
    reason = pd.Series("", index=df.index)

    non_positive = (df[["rv5", "bv5"]] <= 0).any(axis=1) | df[["rv5", "bv5"]].isna().any(axis=1)
    reason[non_positive] = "non-positive or missing rv5/bv5"

    for day, symbol in BAD_DAYS[dataset]:
        reason[(df["date"] == day) & (df["symbol"] == symbol)] = "listed bad day"

    threshold = JUMP_FILTER[dataset]
    if threshold is not None:
        jump_day = (df["rv5"] / df["bv5"] > threshold) & (reason == "")
        reason[jump_day] = f"rv5/bv5 > {threshold:g} (contract roll)"

    masked = reason != ""
    log = df.loc[masked, ["date", "symbol", "rv5", "bv5"]].assign(reason=reason[masked])
    df.loc[masked, MEASURES] = np.nan
    df["masked"] = masked
    return df, log.reset_index(drop=True)


# ------------------------------------------------------------------ 3. features
def add_features(df, week=5, month=22):
    df = df.copy()
    by = df.groupby("symbol", sort=False)

    # target and its log (same definition as the other branches)
    df["rv"] = df["rv5"]
    df["y"] = np.log(df["rv"])
    df["target"] = by["y"].shift(-1)                  # tomorrow's log variance
    df["target_rv"] = by["rv"].shift(-1)              # tomorrow's variance (QLIKE is scored on it)
    df["target_date"] = by["date"].shift(-1)

    # Open-to-close return. It covers the same hours as rv5, and unlike close-to-close
    # returns it is immune to the stock splits left in the price columns.
    df["r"] = np.log(df["close_price"] / df["open_price"])

    # HAR lags, in logs (names of Quentin's script) and in levels
    by_y, by_rv = df.groupby("symbol", sort=False)["y"], df.groupby("symbol", sort=False)["rv"]
    df["x_day"] = df["y"]
    df["x_week"] = by_y.transform(lambda s: s.rolling(week).mean())
    df["x_month"] = by_y.transform(lambda s: s.rolling(month).mean())
    df["rv_day"] = df["rv"]
    df["rv_week"] = by_rv.transform(lambda s: s.rolling(week).mean())
    df["rv_month"] = by_rv.transform(lambda s: s.rolling(month).mean())

    # market-wide level: average log variance of all assets on each date
    market = df.groupby("date")["y"].mean()
    df["m_day"] = df["date"].map(market)
    df["m_week"] = df["date"].map(market.rolling(week).mean())

    # share of today's variance that comes from jumps
    df["jump_share"] = (df["rv5"] - df["bv5"]).clip(lower=0) / df["rv5"]

    # HARQ: the daily lag is trusted less when it is measured with more noise
    df["rv_day_x_sqrt_rq"] = df["rv"] * np.sqrt(df["rq5"])

    # SHAR: good and bad volatility
    df["x_pos"] = np.log(df["rsp5"].where(df["rsp5"] > 0))
    df["x_neg"] = np.log(df["rsn5"].where(df["rsn5"] > 0))
    return df


def build_panel(cfg: RunConfig):
    raw = load_raw(cfg.dataset)
    if cfg.symbols:
        raw = raw[raw["symbol"].isin(cfg.symbols)]
    if cfg.start:
        raw = raw[raw["date"] >= cfg.start]
    if cfg.end:
        raw = raw[raw["date"] <= cfg.end]
    if cfg.quick and not cfg.symbols:
        raw = raw[raw["symbol"].isin(sorted(raw["symbol"].unique())[:3])]
    cleaned, mask_log = clean(raw.reset_index(drop=True), cfg.dataset)
    panel = add_features(cleaned, *cfg.har_lags)
    return panel, mask_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="stocks")
    args = parser.parse_args()
    panel, mask_log = build_panel(RunConfig(dataset=args.dataset))
    print(f"{args.dataset}: {len(panel):,} rows, {panel['symbol'].nunique()} assets, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"masked asset-days: {len(mask_log)}")
    if len(mask_log):
        print(mask_log.to_string(index=False))
    usable = panel[["x_month", "target"]].notna().all(axis=1).sum()
    print(f"rows usable by HAR (22 complete lags and a target): {usable:,}")
