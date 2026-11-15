"""
Look-ahead checks: the split and the features use nothing that is unknown at the forecast
origin. Fails loudly on the first violation.

    python -m ml.checks
"""
import numpy as np
import pandas as pd

from econometrics.config import COMBINED, DATASETS
from econometrics.data import load_raw

from .trees import CUTOFFS, load, split


def check_split(dataset):
    panel, _ = load(dataset)
    for cutoff in CUTOFFS:
        train, test, fit, val = split(panel, cutoff)
        # every training label is known at the first test origin (the close of that day)
        assert train["target_date"].max() <= test["date"].min(), (dataset, cutoff)
        # no test target is ever a training label
        shared = pd.MultiIndex.from_frame(train[["symbol", "target_date"]]).intersection(
            pd.MultiIndex.from_frame(test[["symbol", "target_date"]]))
        assert len(shared) == 0, (dataset, cutoff, len(shared))
        # same rule between the fit part and the validation year
        assert fit["target_date"].max() <= val["date"].min(), (dataset, cutoff)
    return len(panel)


def check_features(dataset="stocks", symbol="AAPL"):
    """Recompute target and HAR lags from the raw file: they must use t and before only."""
    panel, _ = load(dataset)
    one = panel[panel["symbol"] == symbol].set_index("date")
    raw = load_raw(dataset).query("symbol == @symbol").set_index("date")
    y = np.log(raw["rv5"])
    assert np.allclose(one["target"], y.shift(-1).reindex(one.index)), "target != y(t+1)"
    assert np.allclose(one["x_week"], y.rolling(5).mean().reindex(one.index)), "x_week"
    assert np.allclose(one["x_month"], y.rolling(22).mean().reindex(one.index)), "x_month"


if __name__ == "__main__":
    for dataset in [*DATASETS, COMBINED]:
        print(f"{dataset}: split ok on {check_split(dataset):,} rows")
    check_features()
    print("features ok: target = y(t+1), weekly and monthly lags end at t")
