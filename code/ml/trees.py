"""
Tree models for tomorrow's log(rv5): random forest, LightGBM, and LightGBM fitted on the
residual of logHAR-MJ. Linear benchmarks are fitted on the same rows.

    cd code
    python -m ml.trees --dataset stocks           # stocks | forex | futures | combined
    python -m ml.trees --dataset stocks --quick   # 3 assets, only to check the code

Data, cleaning, target, cutoffs and losses come from code/econometrics (imported, not
copied). Fixed split: train before the cutoff, test after. The last year before the
cutoff selects LightGBM's tree size (by QLIKE) and its number of rounds (early stopping);
every model is then refit on the whole training period.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from econometrics import evaluation as ev
from econometrics.config import COMBINED, CUTOFFS, DATASETS, ROOT, RunConfig
from econometrics.data import build_panel

OUT = ROOT / "results" / "ml"
BENCHMARK = "logHAR-MJ"
HAR_MJ = ["x_day", "x_week", "x_month", "m_day", "m_week", "jump_share"]
LEVERAGE = ["r_up", "r_down"]                # yesterday's return split by sign (LHAR)
EXTRA = ["x_pos", "x_neg", "log_rq", "r"]    # semivariances, quarticity, signed return

LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, seed=0, verbose=-1)
GRID = [dict(num_leaves=l, min_child_samples=m) for l in (7, 15, 31) for m in (100, 500)]


# ------------------------------------------------------------------ data
def load(dataset, quick=False):
    """Jean's panel plus a few columns. Returns the usable rows and the tree features."""
    panel, _ = build_panel(RunConfig(dataset=dataset, quick=quick))
    panel["log_rq"] = np.log(panel["rq5"].where(panel["rq5"] > 0))
    panel["r_up"], panel["r_down"] = panel["r"].clip(lower=0), (-panel["r"]).clip(lower=0)
    features = HAR_MJ + EXTRA
    if dataset == COMBINED:
        panel["cls"] = panel["asset_class"].astype("category").cat.codes
        features.append("cls")
    panel = panel.dropna(subset=features + ["target", "target_rv"]).reset_index(drop=True)
    return panel, features


def split(panel, cutoff):
    """train / test around the cutoff; fit / val = train without / with its last year."""
    train, test = panel[panel["date"] < cutoff], panel[panel["date"] >= cutoff]
    in_val = train["date"] >= pd.Timestamp(cutoff) - pd.DateOffset(years=1)
    return train, test, train[~in_val], train[in_val]


# ------------------------------------------------------------------ linear models
def ols(train, test, cols):
    """One regression for all assets. Returns (fitted values on train, forecasts on test)."""
    X = lambda d: np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
    beta = np.linalg.lstsq(X(train), train["target"].to_numpy(), rcond=None)[0]
    return X(train) @ beta, X(test) @ beta


def ols_by_asset(train, test, cols):
    """One regression per asset. Returns (training residuals, forecasts on test)."""
    pred, resid = pd.Series(np.nan, index=test.index), []
    for symbol, tr in train.groupby("symbol"):
        te = test[test["symbol"] == symbol]
        fitted, pred[te.index] = ols(tr, te, cols)
        resid.append(tr["target"] - fitted)
    return pd.concat(resid), pred


# ------------------------------------------------------------------ trees
def boost(fit, val, y_fit, y_val, base_val, features):
    """
    Grid candidate with the lowest validation QLIKE (rounds by early stopping).
    base_val is added back to the prediction before scoring (0, or the HAR forecast).
    Returns (params, rounds, variance of the validation log error, grid table).
    """
    rows = []
    for cand in GRID:
        model = lgb.train({**LGB_PARAMS, **cand}, lgb.Dataset(fit[features], y_fit), 5000,
                          valid_sets=[lgb.Dataset(val[features], y_val)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        err = y_val - model.predict(val[features], num_iteration=model.best_iteration)
        s2 = np.var(err)
        h = np.exp(base_val + y_val - err + s2 / 2)
        rows.append({**cand, "rounds": model.best_iteration, "s2": s2,
                     "val_qlike": ev.qlike(val["target_rv"], h).mean()})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["val_qlike"].idxmin()]
    params = {**LGB_PARAMS, **{k: int(best[k]) for k in GRID[0]}}
    return params, int(best["rounds"]), best["s2"], grid


def har_lgbm(train, test, fit, val, features):
    """LightGBM on the residual of logHAR-MJ: HAR does the linear part, the trees the rest."""
    har_fit, har_val = ols(fit, val, HAR_MJ)
    har_train, har_test = ols(train, test, HAR_MJ)
    params, n, s2, grid = boost(fit, val, fit["target"] - har_fit, val["target"] - har_val,
                                har_val, features)
    model = lgb.train(params, lgb.Dataset(train[features], train["target"] - har_train), n)
    return har_test + model.predict(test[features]), s2, grid, model


def forecast(panel, cutoff, features):
    """Log forecasts on the test rows and, per model, the variance of its log error."""
    train, test, fit, val = split(panel, cutoff)
    pred, s2 = {}, {}

    fitted, pred[BENCHMARK] = ols(train, test, HAR_MJ)
    s2[BENCHMARK] = np.var(train["target"] - fitted)
    for name, cols in [("logHAR-MJ per asset", HAR_MJ), ("LHAR-MJ per asset", HAR_MJ + LEVERAGE)]:
        resid, pred[name] = ols_by_asset(train, test, cols)
        s2[name] = np.var(resid)

    # out-of-bag error: an honest error variance (the in-sample one is close to 0)
    rf = RandomForestRegressor(300, min_samples_leaf=50, max_features=0.5, oob_score=True,
                               n_jobs=-1, random_state=0).fit(train[features], train["target"])
    pred["RF"] = rf.predict(test[features])
    s2["RF"] = np.var(train["target"] - rf.oob_prediction_)

    params, n, s2["LGBM"], grid = boost(fit, val, fit["target"], val["target"], 0, features)
    model = lgb.train(params, lgb.Dataset(train[features], train["target"]), n)
    pred["LGBM"] = model.predict(test[features])
    grids = [grid.assign(model="LGBM")]

    pred["HAR-MJ+LGBM"], s2["HAR-MJ+LGBM"], grid, model = har_lgbm(train, test, fit, val, features)
    grids.append(grid.assign(model="HAR-MJ+LGBM"))
    gain = model.feature_importance("gain")
    importance = pd.DataFrame({"feature": features, "gain_share": gain / gain.sum()})

    return (test, pd.DataFrame(pred, index=test.index), pd.Series(s2),
            pd.concat(grids), importance)


# ------------------------------------------------------------------ scoring
def score(test, logf, s2, reps):
    h = np.exp(logf + s2 / 2)                      # E[RV] when the log error is ~normal
    q, mse_log, mse_rv = ev.loss_tables(test, h, logf)
    daily = ev.daily_mean(q, test["date"])
    dm = [ev.dm_test(daily[m], daily[BENCHMARK]) for m in logf]
    return pd.DataFrame({"n_forecasts": len(test), "qlike": q.mean(),
                         "mse_log_rv": mse_log.mean(), "mse_rv": mse_rv.mean(),
                         "dm_vs_benchmark": [d[0] for d in dm], "dm_pvalue": [d[1] for d in dm],
                         "sigma2": s2}).join(ev.model_confidence_set(daily, reps=reps))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=[*DATASETS, COMBINED], default="stocks")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    panel, features = load(args.dataset, args.quick)
    tables, forecasts, grids, importances = [], [], [], []
    for cutoff in CUTOFFS:
        test, logf, s2, grid, importance = forecast(panel, cutoff, features)
        tables.append(score(test, logf, s2, 200 if args.quick else 2000).assign(cutoff=cutoff))
        forecasts.append(test[["date", "symbol", "target"]].join(logf).assign(cutoff=cutoff))
        grids.append(grid.assign(cutoff=cutoff))
        importances.append(importance.assign(cutoff=cutoff))
        print(f"\n{args.dataset}, test from {cutoff}\n{tables[-1].round(4).to_string()}")

    if not args.quick:
        out = OUT / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).rename_axis("model").to_csv(out / "metrics.csv")
        pd.concat(grids).to_csv(out / "tuning.csv", index=False)
        pd.concat(importances).to_csv(out / "importance.csv", index=False)
        pd.concat(forecasts).to_csv(out / "forecasts.csv.gz", index=False)
        print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
