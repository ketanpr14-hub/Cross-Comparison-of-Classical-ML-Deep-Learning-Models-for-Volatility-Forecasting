"""
Volatility models. Every model returns a one-day-ahead forecast of the realized variance.

Return-based (see only the daily open-to-close return)
    EWMA        RiskMetrics exponential smoothing of squared returns (fixed lambda)
    ARCH        ARCH(p)
    GARCH       GARCH(1,1)
    GJR-GARCH   GARCH(1,1) with a leverage term
    EGARCH      EGARCH(1,1) with a leverage term
Realized-measure based (see the intraday realized variance)
    RW          tomorrow = today
    EWMA-RV     exponential smoothing of realized variance, lambda estimated by QLIKE
    HAR         Corsi (2009), in levels
    HARQ        Bollerslev, Patton & Quaedvlieg (2016): HAR with a measurement-error term
    logHAR      HAR in logs (the "HAR base" of the other branches)
    logHAR-MJ   logHAR + market level + jump share (the "base + market + jump" set)
    logSHAR     Patton & Sheppard (2015): the daily lag split into good and bad volatility
    MEM         multiplicative error model of Engle (2002): a GARCH(1,1) recursion on RV
    AMEM        MEM where negative-return days raise tomorrow's variance more

Parameters are estimated on the training part of each fold only, then kept fixed while
the model is rolled through the test part with the realized data of each new day.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

from .config import RunConfig

RETURN_MODELS = ["EWMA", "ARCH", "GARCH", "GJR-GARCH", "EGARCH"]
ASSET_MODELS = RETURN_MODELS + ["EWMA-RV", "MEM", "AMEM"]      # estimated asset by asset
LOG_MODELS = {                                                  # linear regressions on log RV
    "logHAR": ["x_day", "x_week", "x_month"],
    "logHAR-MJ": ["x_day", "x_week", "x_month", "m_day", "m_week", "jump_share"],
    "logSHAR": ["x_pos", "x_neg", "x_week", "x_month"],
}
LEVEL_MODELS = {                                                # linear regressions on RV
    "HAR": ["rv_day", "rv_week", "rv_month"],
    "HARQ": ["rv_day", "rv_week", "rv_month", "rv_day_x_sqrt_rq"],
}
BURN_IN = 22


# ------------------------------------------------------------------ folds
@dataclass
class Fold:
    train_start: pd.Timestamp     # first training date (inclusive)
    test_start: pd.Timestamp      # first forecast origin (inclusive) = end of training (exclusive)
    test_end: pd.Timestamp        # end of this fold's forecast origins (exclusive)


def make_folds(dates, cutoff, cfg: RunConfig):
    """
    fixed     : one estimation on everything before the cutoff (policy of the other branches)
    expanding : re-estimate every `refit_every` days on all the data available so far
    rolling   : re-estimate every `refit_every` days on the last `window` days
    """
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    cutoff = pd.Timestamp(cutoff)
    first, far = dates[0], dates[-1] + pd.Timedelta(days=1)
    if cfg.split_policy == "fixed":
        return [Fold(first, cutoff, far)]
    if cfg.split_policy not in ("expanding", "rolling"):
        raise ValueError(f"unknown split policy {cfg.split_policy!r}")
    test_dates = dates[dates >= cutoff]
    starts = [cutoff] + list(test_dates[cfg.refit_every::cfg.refit_every])
    folds = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else far
        train_start = first
        if cfg.split_policy == "rolling":
            train_start = dates[max(0, dates.searchsorted(start) - cfg.window)]
        folds.append(Fold(train_start, start, end))
    return folds


# ------------------------------------------------------------------ per-asset models
def qlike(rv, h):
    ratio = rv / h
    return ratio - np.log(ratio) - 1


def _ewma(x, lam, h0):
    """out[t] = forecast for t+1 made at t = lam * out[t-1] + (1 - lam) * x[t]."""
    out, _ = lfilter([1 - lam], [1, -lam], x, zi=[lam * h0])
    return out


def default_hyper(cfg: RunConfig):
    """Hyperparameters used when nothing is tuned (textbook values, set in RunConfig)."""
    orders = {"p": 1, "q": 1}
    regression = {"pooling": cfg.har_pooling}
    return {
        "RW": {}, "EWMA-RV": {},
        "EWMA": {"lambda": cfg.ewma_lambda},
        "ARCH": {"p": cfg.arch_p, "dist": cfg.garch_dist},
        "GARCH": {**orders, "dist": cfg.garch_dist},
        "GJR-GARCH": {**orders, "dist": cfg.garch_dist},
        "EGARCH": {**orders, "dist": cfg.garch_dist},
        "MEM": dict(orders), "AMEM": dict(orders),
        "HAR": {**regression, "estimator": "ols"}, "HARQ": {**regression, "estimator": "ols"},
        "logHAR": dict(regression), "logHAR-MJ": dict(regression), "logSHAR": dict(regression),
    }


def _arch_spec(name, hp):
    if name == "ARCH":
        return dict(vol="ARCH", p=hp["p"], dist=hp["dist"])
    return dict(vol="EGARCH" if name == "EGARCH" else "GARCH", p=hp["p"], q=hp["q"],
                o=0 if name in ("GARCH", "MEM") else 1, dist=hp.get("dist", "normal"))


def _asset_forecasts(symbol, dates, r, rv, folds, specs, cfg):
    """
    All per-asset models for one asset. `dates`, `r`, `rv` hold its valid days only.
    `specs` maps a label to (model name, hyperparameters), so several versions of the
    same model can be run at once when tuning.
    """
    out = {label: np.full(len(rv), np.nan) for label in specs}
    params = []
    for k, fold in enumerate(folds):
        i0, i1, i2 = (int(np.searchsorted(dates, np.datetime64(d)))
                      for d in (fold.train_start, fold.test_start, fold.test_end))
        if i1 - i0 < cfg.min_train or i2 <= i1:
            continue
        for label, (name, hp) in specs.items():
            row = {"model": label, "symbol": symbol, "fold": k, "n_train": i1 - i0}
            if name in ("EWMA", "EWMA-RV"):
                x = (r ** 2 if name == "EWMA" else rv)[i0:i2]
                h0 = x[:BURN_IN].mean()
                lam = hp.get("lambda")
                if name == "EWMA-RV":
                    # choose lambda on the training days, with the loss used for testing
                    def loss(lam):
                        h = _ewma(x[:i1 - i0], lam, h0)
                        return qlike(rv[i0 + BURN_IN + 1:i1], h[BURN_IN:-1]).mean()
                    lam = minimize_scalar(loss, bounds=(0.01, 0.995), method="bounded").x
                out[label][i1:i2] = _ewma(x, lam, h0)[i1 - i0:]
                row.update(converged=True, **{"lambda": lam})
            else:
                # Percent units keep the optimizer well scaled. The MEM is a GARCH on
                # sqrt(RV): its "squared return" is RV itself (Engle & Gallo, 2006).
                if name in RETURN_MODELS:
                    x = 100 * r
                elif name == "MEM":
                    x = 100 * np.sqrt(rv)
                else:
                    x = 100 * np.sqrt(rv) * np.where(r < 0, -1.0, 1.0)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = arch_model(x[:i2], mean="Zero", rescale=False, **_arch_spec(name, hp))
                        fit = model.fit(first_obs=i0, last_obs=i1, disp="off", show_warning=False)
                        forecast = fit.forecast(horizon=1, start=i1, reindex=False)
                    out[label][i1:i2] = forecast.variance.to_numpy()[:, 0] / 1e4
                    row.update(converged=fit.convergence_flag == 0, **fit.params.to_dict())
                except Exception as error:                      # keep the run alive, report it
                    row.update(converged=False, error=str(error)[:200])
            params.append(row)
    return symbol, out, params


def asset_models(panel, folds, specs, cfg):
    """Returns {label: Series of variance forecasts indexed like `panel`} and the parameters."""
    valid = panel[panel["r"].notna() & panel["rv"].notna()]
    jobs = [
        delayed(_asset_forecasts)(symbol, g["date"].to_numpy(), g["r"].to_numpy(),
                                  g["rv"].to_numpy(), folds, specs, cfg)
        for symbol, g in valid.groupby("symbol", sort=False)
    ]
    results = Parallel(n_jobs=cfg.n_jobs)(jobs)
    index = {symbol: g.index for symbol, g in valid.groupby("symbol", sort=False)}
    forecasts = {
        m: pd.concat([pd.Series(out[m], index=index[symbol]) for symbol, out, _ in results])
        for m in specs
    }
    params = [row for _, _, rows in results for row in rows]
    return forecasts, params


# ------------------------------------------------------------------ regressions (HAR family)
def _ols(X, y, weights=None):
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.ones(len(y)) if weights is None else np.sqrt(weights / weights.mean())
    beta = np.linalg.lstsq(X1 * w[:, None], y * w, rcond=None)[0]
    return beta, y - X1 @ beta


def regression_model(panel, folds, name, hp, cfg, label=None):
    """Returns (variance forecast, log forecast or None, parameter rows)."""
    label = label or name
    in_logs = name in LOG_MODELS
    features = (LOG_MODELS if in_logs else LEVEL_MODELS)[name]
    target = "target" if in_logs else "target_rv"
    usable = panel[features + [target]].notna().all(axis=1)
    has_x = panel[features].notna().all(axis=1)
    h = pd.Series(np.nan, index=panel.index)
    logf = pd.Series(np.nan, index=panel.index) if in_logs else None
    params = []
    groups = [("pooled", slice(None))] if hp["pooling"] == "pooled" else \
             [(s, panel["symbol"] == s) for s in panel["symbol"].unique()]
    for k, fold in enumerate(folds):
        in_train = usable & (panel["date"] >= fold.train_start) & (panel["date"] < fold.test_start)
        in_test = has_x & (panel["date"] >= fold.test_start) & (panel["date"] < fold.test_end)
        for group, rows in groups:
            train, test = panel.loc[rows][in_train[rows]], panel.loc[rows][in_test[rows]]
            if len(train) < cfg.min_train or test.empty:
                continue
            # Weighted least squares (levels only): a day with high variance is also a day
            # with a noisy target, so it gets less weight (Clements & Preve, 2021).
            weights = 1 / train["rv_day"].to_numpy() if hp.get("estimator") == "wls" else None
            beta, resid = _ols(train[features].to_numpy(), train[target].to_numpy(), weights)
            pred = beta[0] + test[features].to_numpy() @ beta[1:]
            if in_logs:
                logf.loc[test.index] = pred
                # E[RV] = exp(mean + variance / 2) when the log residual is close to normal
                h.loc[test.index] = np.exp(pred + resid.var() / 2)
            else:
                # "insanity filter": a linear model in levels can go negative or explode, so
                # keep each forecast inside what the asset showed during training
                lo = test["symbol"].map(train.groupby("symbol")["rv"].min()).to_numpy()
                hi = test["symbol"].map(train.groupby("symbol")["rv"].max()).to_numpy()
                h.loc[test.index] = np.clip(pred, lo, hi)
            params.append({"model": label, "symbol": group, "fold": k, "n_train": len(train),
                           "converged": True, "const": beta[0],
                           **dict(zip(features, beta[1:])),
                           **({"resid_var": resid.var()} if in_logs else {})})
    return h, logf, params


# ------------------------------------------------------------------ everything
def run_specs(panel, folds, specs, cfg: RunConfig):
    """
    `specs` maps a label to (model name, hyperparameters).
    Returns variance forecasts H, log forecasts LOGF (both indexed like panel) and parameters.
    """
    H, LOGF, params = {}, {}, []
    per_asset = {label: spec for label, spec in specs.items() if spec[0] in ASSET_MODELS}
    if per_asset:
        forecasts, rows = asset_models(panel, folds, per_asset, cfg)
        H.update(forecasts)
        params += rows
    in_test = (panel["date"] >= folds[0].test_start)
    for label, (name, hp) in specs.items():
        if name == "RW":
            H[label] = panel["rv"].where(in_test)
            LOGF[label] = panel["y"].where(in_test)
        elif name in LOG_MODELS or name in LEVEL_MODELS:
            H[label], logf, rows = regression_model(panel, folds, name, hp, cfg, label)
            params += rows
            if logf is not None:
                LOGF[label] = logf
        elif name not in ASSET_MODELS:
            raise ValueError(f"unknown model {name!r}")
    H = pd.DataFrame(H).reindex(panel.index)[list(specs)]
    # Models that forecast the variance are scored in logs on log(forecast).
    LOGF = pd.DataFrame({m: LOGF[m] if m in LOGF else np.log(H[m]) for m in specs})
    return H, LOGF, pd.DataFrame(params)


def run_models(panel, folds, cfg: RunConfig, hyper=None):
    """Every model of the run, with the given hyperparameters (default: those of RunConfig)."""
    hyper = hyper or default_hyper(cfg)
    return run_specs(panel, folds, {m: (m, hyper[m]) for m in cfg.models}, cfg)
