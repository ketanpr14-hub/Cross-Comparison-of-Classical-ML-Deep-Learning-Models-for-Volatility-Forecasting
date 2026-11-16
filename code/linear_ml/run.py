"""
Ridge, lasso and k-nearest neighbors for tomorrow's log(rv5), on the project's shared protocol.

    cd code
    python -m linear_ml.run --dataset stocks                      # fixed split (main)
    python -m linear_ml.run --dataset stocks --split expanding    # refit every year, all past data
    python -m linear_ml.run --dataset stocks --split rolling      # refit every year, last 1,000 days
    python -m linear_ml.run --dataset forex --pooling per_asset   # one model per asset
    python -m linear_ml.run --dataset stocks --quick              # 3 assets, only to check the code

Data, cleaning, target, cutoffs, folds and losses come from code/econometrics (imported, not
copied), so the test rows are the same as in the other branches.

In every fold, the last year before the fold's first test day is the validation year:
  1. each hyperparameter candidate is fitted on the years before it and scored by QLIKE on it;
  2. the best candidate is refitted on the whole training period and forecasts the fold;
  3. sigma2, the variance of its validation log errors, turns the log forecast into a variance
     forecast: h = exp(log forecast + sigma2 / 2), the rule of the DL branch.
Nothing after a fold's first test day is used to fit, tune or calibrate it.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from econometrics import evaluation as ev
from econometrics.config import COMBINED, CUTOFFS, DATASETS, ROOT, RunConfig
from econometrics.data import build_panel
from econometrics.models import make_folds

OUT = ROOT / "results" / "linear_ml"

# ------------------------------------------------------------------ features
# Column names are those of econometrics.data.build_panel. Everything is known at the close
# of day t; the target is log(rv5) of day t+1.
HAR = ["x_day", "x_week", "x_month"]                     # log RV: day, week, month
EXT = ["x_pos", "x_neg", "jump_share", "log_rq"]         # semivariances, jumps, quarticity
MKT = ["m_day", "m_week"]                                # market-wide log RV (own asset class)

FEATURE_SETS = {
    "har": HAR,
    "har_ext": HAR + EXT,
    "har_mkt": HAR + EXT + MKT,
}
BENCHMARK = "logHAR-MJ"                                  # the reference of every branch
BENCHMARK_FEATURES = HAR + MKT + ["jump_share"]

# ------------------------------------------------------------------ models and grids
# Ridge penalizes the SUM of squared errors, so its alpha grows with the number of rows;
# lasso penalizes the MEAN, so its alpha is in units of the target. Hence the two grids.
GRIDS = {
    "OLS": [{}],
    "Ridge": [{"alpha": float(a)} for a in np.logspace(-2, 5, 15)],
    "Lasso": [{"alpha": float(a)} for a in np.logspace(-5, -1, 13)],
    "kNN-level": [{"k": k, "weights": w} for k in (10, 25, 50, 100, 200) for w in ("uniform", "distance")],
    "kNN-change": [{"k": k, "weights": w} for k in (10, 25, 50, 100, 200) for w in ("uniform", "distance")],
}
MODELS = list(GRIDS)

# kNN-change forecasts tomorrow = today + the neighbors' average change. Its forecast can go
# beyond anything seen in training (e.g. March 2020); kNN-level cannot.
CHANGE_BASE = "x_day"


def make_estimator(model, params):
    if model == "OLS":
        return LinearRegression()
    if model == "Ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=params["alpha"]))])
    if model == "Lasso":
        return Pipeline([("scale", StandardScaler()),
                         ("model", Lasso(alpha=params["alpha"], max_iter=20_000))])
    if model.startswith("kNN"):
        return Pipeline([("scale", StandardScaler()),
                         ("model", KNeighborsRegressor(n_neighbors=params["k"],
                                                       weights=params["weights"], n_jobs=-1))])
    raise ValueError(model)


# ------------------------------------------------------------------ data
def load(dataset, quick=False):
    """Jean's panel, restricted to rows where every feature set and the target exist,
    so every model is scored on exactly the same rows."""
    panel, _ = build_panel(RunConfig(dataset=dataset, quick=quick))
    panel["log_rq"] = np.log(panel["rq5"].where(panel["rq5"] > 0))
    needed = sorted(set(sum(FEATURE_SETS.values(), [])) | set(BENCHMARK_FEATURES))
    return panel.dropna(subset=needed + ["target", "target_rv"]).reset_index(drop=True)


# ------------------------------------------------------------------ fit / predict
def fit_predict(model, params, train, test, cols, pooling, min_rows):
    """Log forecasts for `test`, and the fitted estimators (pooled: one; per asset: one each)."""
    base_tr = train[CHANGE_BASE].to_numpy() if model == "kNN-change" else 0.0
    base_te = test[CHANGE_BASE].to_numpy() if model == "kNN-change" else 0.0
    y = train["target"].to_numpy() - base_tr

    if pooling == "pooled":
        est = make_estimator(model, params).fit(train[cols], y)
        return est.predict(test[cols]) + base_te, {"all": est}

    pred, fitted = np.full(len(test), np.nan), {}
    y = pd.Series(y, index=train.index)
    for symbol, tr in train.groupby("symbol"):
        rows = (test["symbol"] == symbol).to_numpy()
        if len(tr) < max(min_rows, params.get("k", 0) + 1) or not rows.any():
            continue
        est = make_estimator(model, params).fit(tr[cols], y.loc[tr.index])
        pred[rows] = est.predict(test.loc[rows, cols])
        fitted[symbol] = est
    return pred + base_te, fitted


def coefficients(model, estimators, cols, fold, label):
    """Standardized coefficients of the linear models, to see which features lasso keeps."""
    if model not in ("Ridge", "Lasso"):
        return []
    return [{"model": label, "fold": fold, "group": g, **dict(zip(cols, est["model"].coef_))}
            for g, est in estimators.items()]


# ------------------------------------------------------------------ one fold
def run_fold(panel, fold, k, specs, pooling, min_rows):
    """Forecasts of every model on the fold's test rows, plus tuning and coefficient rows."""
    dates = panel["date"]
    train = panel[(dates >= fold.train_start) & (dates < fold.test_start)]
    test = panel[(dates >= fold.test_start) & (dates < fold.test_end)]
    val_start = fold.test_start - pd.DateOffset(years=1)
    fit, val = train[train["date"] < val_start], train[train["date"] >= val_start]

    logf, h = pd.DataFrame(index=test.index), pd.DataFrame(index=test.index)
    tuning, coefs = [], []

    # tomorrow = today: nothing to fit; sigma2 from the validation year like everything else
    s2 = np.var(val["target"] - val["x_day"])
    logf["RW"] = test["x_day"]
    h["RW"] = np.exp(test["x_day"] + s2 / 2)

    for label, (model, cols) in specs.items():
        rows = []
        for params in GRIDS[model]:
            pv, _ = fit_predict(model, params, fit, val, cols, pooling, min_rows)
            ok = ~np.isnan(pv)
            err = val["target"].to_numpy()[ok] - pv[ok]
            s2 = np.var(err)
            vq = ev.qlike(val["target_rv"].to_numpy()[ok], np.exp(pv[ok] + s2 / 2)).mean()
            rows.append({"model": label, "fold": k, "params": params, "val_qlike": vq, "sigma2": s2})
        best = min(rows, key=lambda r: r["val_qlike"])
        for r in rows:
            tuning.append({**r, "params": str(r["params"]), "selected": r is best})

        pred, ests = fit_predict(model, best["params"], train, test, cols, pooling, min_rows)
        logf[label] = pred
        h[label] = np.exp(pred + best["sigma2"] / 2)
        coefs += coefficients(model, ests, cols, k, label)

    return test, logf, h, tuning, coefs


def build_specs(models, feature_sets):
    """label -> (model, columns). The benchmark is the group's logHAR-MJ, fitted by OLS."""
    specs = {BENCHMARK: ("OLS", BENCHMARK_FEATURES)}
    for model in models:
        for fs in feature_sets:
            specs[f"{model} [{fs}]"] = (model, FEATURE_SETS[fs])
    return specs


# ------------------------------------------------------------------ scoring
def score(test, logf, h, reps):
    """Same losses and tests as the other branches, on the daily loss averaged across assets."""
    q, mse_log, mse_rv = ev.loss_tables(test, h, logf)
    daily = ev.daily_mean(q, test["date"])
    dm = {m: ev.dm_test(daily[m], daily[BENCHMARK]) for m in logf}
    table = pd.DataFrame({
        "n_forecasts": logf.notna().sum(),
        "qlike": q.mean(),
        "mse_log_rv": mse_log.mean(),
        "mse_rv": mse_rv.mean(),
        "vs_benchmark_pct": 100 * (q.mean() / q[BENCHMARK].mean() - 1),
        "dm_vs_benchmark": {m: d[0] for m, d in dm.items()},
        "dm_pvalue": {m: d[1] for m, d in dm.items()},
    })
    return table.join(ev.model_confidence_set(daily, reps=reps)).sort_values("qlike")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=[*DATASETS, COMBINED], default="stocks")
    p.add_argument("--split", choices=["fixed", "expanding", "rolling"], default="fixed")
    p.add_argument("--window", type=int, default=1000, help="training days for --split rolling")
    p.add_argument("--refit-every", type=int, default=252, help="days between refits (expanding/rolling)")
    p.add_argument("--pooling", choices=["pooled", "per_asset"], default="pooled")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS), default=list(FEATURE_SETS))
    p.add_argument("--quick", action="store_true", help="3 assets, few bootstrap draws")
    args = p.parse_args()

    cfg = RunConfig(dataset=args.dataset, split_policy=args.split, window=args.window,
                    refit_every=args.refit_every, quick=args.quick)
    panel = load(args.dataset, args.quick)
    specs = build_specs(args.models, args.feature_sets)
    print(f"{args.dataset}: {len(panel):,} usable rows, {panel['symbol'].nunique()} assets, "
          f"{len(specs) + 1} models, {args.split} split, {args.pooling}")

    tables, forecasts, tunings, coefs = [], [], [], []
    for cutoff in CUTOFFS:
        folds = make_folds(panel["date"], cutoff, cfg)
        parts = []
        for k, fold in enumerate(folds):
            t0 = time.time()
            parts.append(run_fold(panel, fold, k, specs, args.pooling, cfg.min_train))
            tunings += [dict(r, cutoff=cutoff) for r in parts[-1][3]]
            coefs += [dict(r, cutoff=cutoff) for r in parts[-1][4]]
            print(f"  cutoff {cutoff} fold {k + 1}/{len(folds)}: test from "
                  f"{fold.test_start.date()}, {len(parts[-1][0]):,} rows, {time.time() - t0:.0f}s",
                  flush=True)
        test = pd.concat([pt[0] for pt in parts])
        logf = pd.concat([pt[1] for pt in parts])
        h = pd.concat([pt[2] for pt in parts])
        tables.append(score(test, logf, h, 200 if args.quick else 2000).assign(cutoff=cutoff))
        forecasts.append(test[["date", "symbol", "target"]].join(logf).assign(cutoff=cutoff))
        print(f"\n{args.dataset}, {args.split}, test from {cutoff}\n"
              f"{tables[-1].drop(columns='cutoff').round(4).to_string()}\n")

    if not args.quick:
        suffix = "" if args.pooling == "pooled" else "_per_asset"
        out = OUT / args.dataset / f"{args.split}{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).rename_axis("model").to_csv(out / "metrics.csv")
        pd.DataFrame(tunings).to_csv(out / "tuning.csv", index=False)
        pd.DataFrame(coefs).to_csv(out / "coefficients.csv", index=False)
        pd.concat(forecasts).to_csv(out / "forecasts.csv.gz", index=False)   # not versioned
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
