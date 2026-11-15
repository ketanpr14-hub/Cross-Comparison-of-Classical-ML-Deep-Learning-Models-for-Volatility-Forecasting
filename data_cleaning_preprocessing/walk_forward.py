"""
Rolling-window walk-forward splits.

- walk_forward_generator(): yields (train_idx, test_idx) for one symbol's feature table.
  Import this in the model scripts.
- main(): saves a table of fold dates per symbol, for checking and for the report.

Reads:  data/model_data/{fx,futures,stocks}_model_data/<symbol>.parquet
Writes: data/walk_forward/{fx,futures,stocks}_walk_forward/<symbol>.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import (ASSET_CLASSES, DATA_DIR, GAP, MODEL_DATA_DIR, OOS_START, STEP_SIZE,
                    TEST_SIZE, TRAIN_SIZE)

WF_DIR = DATA_DIR / "walk_forward"


def walk_forward_generator(data, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                           step_size=STEP_SIZE, gap=GAP, oos_start=OOS_START):
    """
    data: one symbol's feature table with a 'date' column, sorted by date.
    Yields (train_idx, test_idx) as positional row indices.

    Layout of each fold:
      [ train_size rows ][ gap rows, unused ][ test_size rows ]
    """
    dates = pd.to_datetime(data["date"]).reset_index(drop=True)
    n = len(dates)

    test_start = int(dates.searchsorted(oos_start))  # first row on or after oos_start
    first_train_start = test_start - gap - train_size
    if first_train_start < 0:
        raise ValueError(
            f"Only {test_start} rows before {oos_start.date()}, but train_size + gap "
            f"= {train_size + gap}. Lower train_size or start the data earlier."
        )

    while test_start < n:
        test_end = min(test_start + test_size, n)   # last block may be shorter
        train_end = test_start - gap
        train_idx = np.arange(train_end - train_size, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
        test_start += step_size


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fold with its train and test date ranges."""
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    rows = []
    for fold, (tr, te) in enumerate(walk_forward_generator(df)):
        rows.append({
            "fold": fold,
            "train_start": dates[tr[0]], "train_end": dates[tr[-1]],
            "test_start": dates[te[0]], "test_end": dates[te[-1]],
            "n_train": len(tr), "n_test": len(te),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for asset_class in ASSET_CLASSES:
        in_dir = MODEL_DATA_DIR / f"{asset_class}_model_data"
        out_dir = WF_DIR / f"{asset_class}_walk_forward"
        out_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(in_dir.glob("*.parquet")):
            df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
            folds = fold_table(df)
            folds.to_parquet(out_dir / path.name, index=False)

            print(f"{asset_class:8s} {path.stem:12s} {len(folds):4d} folds  "
                  f"first test {folds['test_start'].iloc[0].date()}  "
                  f"last test {folds['test_end'].iloc[-1].date()}")


if __name__ == "__main__":
    main()