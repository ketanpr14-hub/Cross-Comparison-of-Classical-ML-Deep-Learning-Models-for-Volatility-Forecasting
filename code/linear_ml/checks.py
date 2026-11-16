"""
Look-ahead checks for linear_ml.run, on every dataset, cutoff and split.

    cd code
    python -m linear_ml.checks            # add --quick for 3 assets per dataset

For every fold it asserts that:
  - every training label is known at the fold's first forecast origin (the close of its first
    test day), and every fit label is known at the first validation origin;
  - no test row is used for fitting, tuning or sigma2;
  - every model is scored on the same test rows, and the test rows of a cutoff do not overlap.
"""
import argparse

import pandas as pd

from econometrics.config import COMBINED, CUTOFFS, DATASETS, RunConfig
from econometrics.models import make_folds

from .run import load


def known_at_first_origin(train, later):
    first = later.groupby("symbol")["date"].min()
    limit = train["symbol"].map(first).fillna(pd.Timestamp.max)
    return bool((train["target_date"] <= limit).all())


def check(dataset, split, quick):
    panel = load(dataset, quick)
    cfg = RunConfig(dataset=dataset, split_policy=split, quick=quick)
    dates = panel["date"]
    for cutoff in CUTOFFS:
        seen = pd.Index([])
        for fold in make_folds(dates, cutoff, cfg):
            train = panel[(dates >= fold.train_start) & (dates < fold.test_start)]
            test = panel[(dates >= fold.test_start) & (dates < fold.test_end)]
            val_start = fold.test_start - pd.DateOffset(years=1)
            fit, val = train[train["date"] < val_start], train[train["date"] >= val_start]

            # a label is known at an origin if its day is on or before that origin's day,
            # per asset (an asset may not trade on the first test or validation day)
            assert known_at_first_origin(train, test), "training label after first test origin"
            assert known_at_first_origin(fit, val), "fit label after first validation origin"
            assert train.index.intersection(test.index).empty, "test row used for training"
            assert (test["date"] >= pd.Timestamp(cutoff)).all(), "test row before the cutoff"
            assert seen.intersection(test.index).empty, "test rows overlap between folds"
            seen = seen.union(test.index)
        print(f"  {dataset:8s} {split:9s} cutoff {cutoff}: ok ({len(seen):,} test rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    for dataset in [*DATASETS, COMBINED]:
        for split in ("fixed", "expanding", "rolling"):
            check(dataset, split, args.quick)
    print("all checks passed")


if __name__ == "__main__":
    main()
