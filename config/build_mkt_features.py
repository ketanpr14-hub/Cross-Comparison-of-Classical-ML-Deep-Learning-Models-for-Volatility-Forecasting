"""
Add market-average and other-symbol volatility features to every symbol's feature table.

Reads:  data/features/{asset}_features/<symbol>.parquet     (from build_features.py)
Writes: data/model_data/{asset}_model_data/<symbol>.parquet  (what the model scripts read)

For each symbol, within its own asset class:
  log_rv_mkt_d/w/m          average of the OTHER symbols' log_rv_d/w/m on that date
  other_log_rv_d_<SYMBOL>   each other symbol's log_rv_d on that date

All values are from date t, the same day as the symbol's own features, so nothing
from the future is used. Rows missing any feature (of any set) are dropped, so every
feature set and every model is evaluated on exactly the same rows.
"""

import pandas as pd

from config import (ASSET_CLASSES, FEATURE_DIR, FEATURE_SETS, FFILL_LIMIT, MKT_FEATURES,
                    MKT_SOURCE_TERMS, MODEL_DATA_DIR, OTHER_PREFIX, OTHER_SYMBOL_TERMS,
                    feature_columns)


def wide_panel(long: pd.DataFrame, term: str) -> pd.DataFrame:
    """Dates x symbols table of one feature, gaps carried forward a few days."""
    return (long.pivot(index="date", columns="symbol", values=term)
                .sort_index()
                .ffill(limit=FFILL_LIMIT))


def add_cross_features(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    long = pd.concat(frames.values(), ignore_index=True)
    symbols = sorted(frames)
    out = {sym: df.copy() for sym, df in frames.items()}

    # Leave-one-out market averages
    for term, new_col in zip(MKT_SOURCE_TERMS, MKT_FEATURES):
        wide = wide_panel(long, term)
        total, count = wide.sum(axis=1, min_count=1), wide.count(axis=1)
        for sym in symbols:
            loo = (total - wide[sym]) / (count - 1)      # NaN when no other symbol has data
            out[sym][new_col] = out[sym]["date"].map(loo)

    # Other symbols' volatility, one column each
    for term in OTHER_SYMBOL_TERMS:
        wide = wide_panel(long, term)
        for sym in symbols:
            for other in symbols:
                if other != sym:
                    out[sym][f"{OTHER_PREFIX}{term}_{other}"] = out[sym]["date"].map(wide[other])

    return out


def main() -> None:
    for asset_class in ASSET_CLASSES:
        in_dir = FEATURE_DIR / f"{asset_class}_features"
        out_dir = MODEL_DATA_DIR / f"{asset_class}_model_data"
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = {p.stem: pd.read_parquet(p).sort_values("date").reset_index(drop=True)
                  for p in sorted(in_dir.glob("*.parquet"))}
        if len(frames) < 2:
            print(f"{asset_class}: fewer than 2 symbols, market features not possible; skipping")
            continue

        for sym, df in add_cross_features(frames).items():
            # Keep only rows where every feature set is complete, so all sets share the same rows
            needed = sorted({c for s in FEATURE_SETS for c in feature_columns(s, df.columns)} | {"y"})
            before = len(df)
            df = df.dropna(subset=needed).reset_index(drop=True)
            df.to_parquet(out_dir / f"{sym}.parquet", index=False)

            n_other = sum(c.startswith(OTHER_PREFIX) for c in df.columns)
            print(f"{asset_class:8s} {sym:12s} {len(df):6d} rows "
                  f"({before - len(df)} dropped)  {n_other} other-symbol columns")


if __name__ == "__main__":
    main()