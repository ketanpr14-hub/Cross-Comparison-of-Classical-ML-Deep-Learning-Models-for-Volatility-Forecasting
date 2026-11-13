"""
Testing: losses, Diebold-Mariano tests and the Model Confidence Set.

QLIKE is a loss: lower is better, 0 is a perfect forecast.

    QLIKE = RV / h - log(RV / h) - 1        (h = forecast of the variance)

It is the main criterion because it ranks forecasts correctly even though RV is only a
noisy measure of the true variance (Patton, 2011), and it does not let a few crisis days
decide the ranking the way a squared error on the variance does.

The panel has several assets per day and their losses move together, so every test is
run on the daily loss averaged across assets (one number per day and per model).
"""
import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats


def qlike(rv, h):
    ratio = rv / h
    return ratio - np.log(ratio) - 1


def loss_tables(sample, H, LOGF):
    """Per asset-day losses. `sample` has the target columns; H, LOGF the forecasts."""
    q = H.apply(lambda h: qlike(sample["target_rv"], h))
    mse_log = LOGF.apply(lambda f: (sample["target"] - f) ** 2)
    return q, mse_log


def daily_mean(losses, dates):
    """Average across assets: one row per day, one column per model."""
    return losses.groupby(dates.to_numpy()).mean()


# ------------------------------------------------------------------ Diebold-Mariano
def dm_test(loss_a, loss_b, lags=None):
    """
    H0: both forecasts have the same expected loss. Negative statistic = A is better.
    The variance of the mean loss difference is a Newey-West (Bartlett) estimate.
    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 30 or np.allclose(d, 0):
        return np.nan, np.nan
    lags = int(np.floor(n ** (1 / 3))) if lags is None else lags
    u = d - d.mean()
    variance = u @ u / n
    for k in range(1, lags + 1):
        variance += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    if variance <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(variance / n)
    return stat, 2 * stats.norm.sf(abs(stat))


def dm_matrix(daily, lags=None):
    """All pairs, long format: one row per ordered pair (model_a, model_b)."""
    rows = []
    for a in daily.columns:
        for b in daily.columns:
            if a != b:
                stat, p = dm_test(daily[a], daily[b], lags)
                rows.append({"model_a": a, "model_b": b, "dm_stat": stat, "p_value": p,
                             "mean_loss_diff": (daily[a] - daily[b]).mean()})
    return pd.DataFrame(rows)


def dm_by_asset(losses, sample, benchmark, lags=None, level=0.05):
    """For each model: share of assets where it beats / loses to the benchmark significantly."""
    counts = {m: [0, 0] for m in losses.columns if m != benchmark}
    symbols = sample["symbol"].unique()
    for symbol in symbols:
        one = losses[(sample["symbol"] == symbol).to_numpy()]
        for m in counts:
            stat, p = dm_test(one[m], one[benchmark], lags)
            if p < level:
                counts[m][0 if stat < 0 else 1] += 1
    table = pd.DataFrame(counts, index=["assets_sig_better", "assets_sig_worse"]).T
    table["n_assets"] = len(symbols)
    return table


# ------------------------------------------------------------------ Model Confidence Set
def model_confidence_set(daily, size=0.10, reps=2000, block=10, seed=0):
    """
    Hansen, Lunde & Nason (2011). The set contains the best model with probability
    1 - size; a model is in it when its MCS p-value is above `size`.
    """
    mcs = MCS(daily, size=size, reps=reps, block_size=block, method="R",
              bootstrap="stationary", seed=seed)
    mcs.compute()
    p = mcs.pvalues["Pvalue"]
    return pd.DataFrame({"mcs_pvalue": p, "in_mcs": p > size}).rename_axis("model")
