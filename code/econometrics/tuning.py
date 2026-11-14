"""
Hyperparameter tuning on validation years taken from the training period.

For a test cutoff C and `tune_years` = 2:

    block 1   estimate on [start, C - 2y)   validate on [C - 2y, C - 1y)
    block 2   estimate on [start, C - 1y)   validate on [C - 1y, C)

Every candidate of a model is estimated on the first part of each block and scored on its
validation year with QLIKE, the loss used for testing. The candidate with the lowest
QLIKE over all validation asset-days is kept, and the model is then re-estimated with it
on the whole training period. Nothing after the cutoff is ever used.

This is the split of the DL branch (the last year before the cutoff is the validation
set) with two differences: two validation years instead of one, so a choice does not
depend on a single year, and a final re-estimation on all the training data.

One set of hyperparameters is chosen per model and per cutoff, for all assets together.
"""
import json
from itertools import product

import numpy as np
import pandas as pd

from .config import RunConfig
from .models import Fold, default_hyper, qlike, run_specs


def _grid(**options):
    return [dict(zip(options, values)) for values in product(*options.values())]


GRIDS = {
    "EWMA": _grid(**{"lambda": [0.80, 0.85, 0.90, 0.94, 0.97, 0.99]}),
    "ARCH": _grid(p=[1, 3, 5, 10, 22], dist=["normal", "t"]),
    "GARCH": _grid(p=[1, 2], q=[1, 2], dist=["normal", "t"]),
    "GJR-GARCH": _grid(p=[1], q=[1, 2], dist=["normal", "t"]),
    "EGARCH": _grid(p=[1], q=[1, 2], dist=["normal", "t"]),
    "MEM": _grid(p=[1, 2], q=[1, 2]),
    "AMEM": _grid(p=[1, 2], q=[1, 2]),
    "HAR": _grid(pooling=["pooled", "per_asset"], estimator=["ols", "wls"]),
    "HARQ": _grid(pooling=["pooled", "per_asset"], estimator=["ols", "wls"]),
    "logHAR": _grid(pooling=["pooled", "per_asset"]),
    "logHAR-MJ": _grid(pooling=["pooled", "per_asset"]),
    "logSHAR": _grid(pooling=["pooled", "per_asset"]),
    # RW has nothing to tune; EWMA-RV already fits its decay by QLIKE on the training days.
}


def validation_folds(panel, cutoff, cfg: RunConfig):
    cutoff = pd.Timestamp(cutoff)
    first = panel["date"].min()
    edges = [cutoff - pd.DateOffset(years=k) for k in range(cfg.tune_years, -1, -1)]
    return [Fold(first, start, end) for start, end in zip(edges[:-1], edges[1:])]


def tune(panel, cutoff, cfg: RunConfig):
    """Returns ({model: selected hyperparameters}, table of every candidate and its score)."""
    hyper = default_hyper(cfg)
    folds = validation_folds(panel, cutoff, cfg)
    specs, owner = {}, {}
    for model in cfg.models:
        for i, candidate in enumerate(GRIDS.get(model, [])):
            label = f"{model}#{i}"
            specs[label] = (model, candidate)
            owner[label] = model
    if not specs:
        return hyper, pd.DataFrame()

    # Only days before the cutoff are given to the models, and a validation row counts
    # only if the day it forecasts is itself before the cutoff.
    past = panel[panel["date"] < pd.Timestamp(cutoff)]
    H, _, _ = run_specs(past, folds, specs, cfg)
    in_validation = (past["date"] >= folds[0].test_start) & (past["target_date"] < pd.Timestamp(cutoff)) \
        & past["target_rv"].notna()

    rows = []
    for model in cfg.models:
        labels = [label for label in specs if owner[label] == model]
        if not labels:
            continue
        forecasts = H.loc[in_validation, labels]
        # a candidate that cannot be estimated on the short validation samples is dropped
        coverage = forecasts.notna().mean()
        usable = [label for label in labels if coverage[label] >= 0.99 * coverage.max() > 0]
        common = forecasts[usable].dropna()
        scores = common.apply(lambda h: qlike(past.loc[common.index, "target_rv"], h).mean())
        best = scores.idxmin()
        hyper[model] = specs[best][1]
        for label in labels:
            rows.append({
                "cutoff": cutoff, "model": model, "hyperparameters": json.dumps(specs[label][1]),
                "validation_qlike": scores.get(label, np.nan), "n_validation": len(common),
                "is_default": specs[label][1] == default_hyper(cfg)[model], "selected": label == best,
            })
    return hyper, pd.DataFrame(rows)
