"""
Where the gain of HAR-MJ+LGBM comes from. Run after ml.trees on the same dataset:

    python -m ml.diagnose --dataset futures

Writes to results/ml/<dataset>/:
    per_asset.csv   QLIKE of every model on every asset, and its change against logHAR-MJ
    leverage.csv    error of logHAR-MJ explained by yesterday's return, up and down moves apart
    ablation.csv    test QLIKE of HAR-MJ+LGBM when feature groups are added one at a time
    plain.csv       accuracy in plain words: R2, typical error on volatility, hit rates
"""
import argparse

import numpy as np
import pandas as pd

from econometrics.evaluation import qlike

from .trees import BENCHMARK, CUTOFFS, HAR_MJ, OUT, har_lgbm, load, ols, split


def variance_forecasts(f, metrics):
    """h = exp(log forecast + sigma2 / 2), sigma2 as estimated in ml.trees (never on test)."""
    models = metrics.index.unique()
    s2 = metrics.reset_index().set_index(["cutoff", "model"])["sigma2"]
    return {m: np.exp(f[m] + f["cutoff"].map(lambda c: s2[(c, m)]) / 2) for m in models}


def per_asset(f, h):
    q = pd.DataFrame({m: qlike(np.exp(f["target"]), hm) for m, hm in h.items()})
    table = q.groupby(f["symbol"]).mean()
    change = 100 * (table.div(table[BENCHMARK], axis=0) - 1)
    return table.join(change.drop(columns=BENCHMARK).add_suffix(" vs HAR %"))


def leverage(panel):
    """Per asset and cutoff: residual of the pooled logHAR-MJ on [1, r_up, r_down] / sd(r)."""
    rows = []
    for cutoff in CUTOFFS:
        train = split(panel, cutoff)[0]
        resid = train["target"] - ols(train, train, HAR_MJ)[0]
        for symbol, g in train.groupby("symbol"):
            sd = g["r"].std()
            X = np.column_stack([np.ones(len(g)), g["r_up"] / sd, g["r_down"] / sd])
            b = np.linalg.lstsq(X, resid[g.index], rcond=None)[0]
            rows.append({"cutoff": cutoff, "symbol": symbol, "after_up_move": b[1],
                         "after_down_move": b[2]})
    return pd.DataFrame(rows)


def ablation(panel, features):
    groups = {"HAR-MJ only": [], "+ r": ["r"], "+ log_rq": ["log_rq"],
              "+ x_pos, x_neg": ["x_pos", "x_neg"],
              "all features": [c for c in features if c not in HAR_MJ]}
    rows = []
    for cutoff in CUTOFFS:
        train, test, fit, val = split(panel, cutoff)
        for name, extra in groups.items():
            pred, s2, _, _ = har_lgbm(train, test, fit, val, HAR_MJ + extra)
            rows.append({"cutoff": cutoff, "features": name,
                         "qlike": qlike(test["target_rv"], np.exp(pred + s2 / 2)).mean()})
    return pd.DataFrame(rows).pivot(index="features", columns="cutoff", values="qlike")


def plain(f, h):
    """Accuracy in plain words, per cutoff. RW (tomorrow = today) is the naive reference."""
    rows = []
    for m in ["RW", *h]:
        for cutoff, g in f.groupby("cutoff"):
            pred = g["x_day"] if m == "RW" else g[m]
            err = pred - g["target"]                         # error on log variance
            vol_err = np.abs(np.exp(err / 2) - 1)            # relative error on volatility
            moved = np.sign(g["target"] - g["x_day"])
            rows.append({"model": m, "cutoff": cutoff,
                         "r2_log_rv": 1 - (err ** 2).sum() / ((g["target"] - g["target"].mean()) ** 2).sum(),
                         "median_vol_error_pct": 100 * vol_err.median(),
                         "within_20pct_share": 100 * (vol_err < 0.2).mean(),
                         "direction_hit_pct": np.nan if m == "RW"
                         else 100 * (np.sign(pred - g["x_day"]) == moved).mean()})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["stocks", "forex", "futures"], default="futures")
    args = p.parse_args()
    out = OUT / args.dataset

    panel, features = load(args.dataset)
    metrics = pd.read_csv(out / "metrics.csv", index_col="model", dtype={"cutoff": str})
    f = pd.read_csv(out / "forecasts.csv.gz", parse_dates=["date"], dtype={"cutoff": str})
    f = f.merge(panel[["date", "symbol", "x_day"]], on=["date", "symbol"])
    h = variance_forecasts(f, metrics)

    results = {"per_asset": per_asset(f, h), "leverage": leverage(panel),
               "ablation": ablation(panel, features), "plain": plain(f, h)}
    for name, table in results.items():
        table.to_csv(out / f"{name}.csv", index=name in ("per_asset", "ablation"))
        print(f"\n{name}\n{table.round(3).to_string()}")


if __name__ == "__main__":
    main()
