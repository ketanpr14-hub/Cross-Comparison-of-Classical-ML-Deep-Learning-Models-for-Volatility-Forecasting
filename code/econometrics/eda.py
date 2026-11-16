"""
Exploratory analysis of one dataset, aimed at the modelling choices of the pipeline.

    python -m econometrics.eda --dataset stocks        # or forex, futures, all

Writes to results/econometrics/eda/<dataset>/
    summary_by_asset.csv     level, dispersion and memory of volatility, asset by asset
    masked_days.csv          asset-days removed by the cleaning step, with the reason
    volatility_level.png     median annualized volatility through time, with the test cutoffs
    distribution.png         RV is heavily skewed, log RV is close to normal
    memory.png               autocorrelation of log RV versus squared returns
"""
import argparse

import numpy as np
import pandas as pd

from .config import CUTOFFS, DATASETS, RESULTS_DIR, RunConfig
from .data import add_features, clean, load_raw
from .plots import BLUE, GREY, ORANGE, TEXT_2, plt, save, use_style

LAGS = 60


def autocorrelation(series, lags=LAGS):
    x = series.dropna().to_numpy()
    x = x - x.mean()
    return np.array([x[k:] @ x[:-k] / (x @ x) for k in range(1, lags + 1)])


def summarize(raw, panel):
    """One row per asset. Return columns come from the raw prices to show the split issue."""
    raw = raw.assign(r_oc=np.log(raw["close_price"] / raw["open_price"]),
                     r_cc=raw.groupby("symbol")["close_price"].transform(lambda s: np.log(s).diff()))
    rows = []
    for symbol, g in panel.groupby("symbol"):
        raw_g = raw[raw["symbol"] == symbol]
        rows.append({
            "symbol": symbol, "n_days": len(g), "first": g["date"].min().date(), "last": g["date"].max().date(),
            "masked_days": int(g["masked"].sum()),
            "ann_vol_pct": 100 * np.sqrt(252 * g["rv"].mean()),
            "rv_skew": g["rv"].skew(), "rv_kurt": g["rv"].kurt(),
            "log_rv_skew": g["y"].skew(), "log_rv_kurt": g["y"].kurt(),
            "log_rv_acf1": autocorrelation(g["y"], 22)[0], "log_rv_acf22": autocorrelation(g["y"], 22)[21],
            "sq_return_acf1": autocorrelation(g["r"] ** 2, 1)[0],
            "jump_share_mean": g["jump_share"].mean(),
            # which daily return is on the same scale as rv5? (1 = same average variance)
            "open_close_var_over_rv": (raw_g["r_oc"] ** 2).mean() / raw_g["rv5"].mean(),
            "close_close_var_over_rv": (raw_g["r_cc"] ** 2).mean() / raw_g["rv5"].mean(),
            # leverage: does a negative return today come with higher variance tomorrow?
            "corr_return_next_log_rv_change": g["r"].corr(g["target"] - g["y"]),
        })
    return pd.DataFrame(rows)


def run(dataset):
    out = RESULTS_DIR / "eda" / dataset
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw(dataset)
    cleaned, mask_log = clean(raw, dataset)
    cfg = RunConfig(dataset=dataset)
    panel = add_features(cleaned, *cfg.har_lags)

    summary = summarize(raw, panel)
    summary.to_csv(out / "summary_by_asset.csv", index=False, float_format="%.4f")
    mask_log.to_csv(out / "masked_days.csv", index=False)

    print(f"\n===== {dataset}: {len(panel):,} rows, {panel['symbol'].nunique()} assets, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"missing values in the raw file: {int(raw.isna().sum().drop('index', errors='ignore').sum())}"
          f" | duplicated asset-days: {int(raw.duplicated(['symbol', 'date']).sum())}"
          f" | masked asset-days: {len(mask_log)}")
    numeric = summary.drop(columns=["symbol", "first", "last"])
    print(numeric.describe().loc[["mean", "min", "50%", "max"]].T.round(3).to_string())

    use_style()
    # 1. level of volatility through time
    vol = 100 * np.sqrt(252 * panel.groupby("date")["rv"].median())
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(vol.index, vol.rolling(5, min_periods=1).mean(), color=BLUE)
    for cutoff in CUTOFFS:
        ax.axvline(pd.Timestamp(cutoff), color=GREY, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"test from {cutoff[:4]}", (pd.Timestamp(cutoff), ax.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", color=TEXT_2, fontsize=8)
    ax.set_title(f"{dataset}: median annualized volatility across assets (5-day average)")
    ax.set_ylabel("% per year")
    save(fig, out / "volatility_level.png")

    # 2. distribution of the target, in levels and in logs (standardized asset by asset)
    def standardize(col):
        return panel.groupby("symbol")[col].transform(lambda s: (s - s.mean()) / s.std()).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    grid = np.linspace(-4, 6, 200)
    for ax, col, title in zip(axes, ["rv", "y"], ["Realized variance", "Log realized variance"]):
        z = standardize(col)
        ax.hist(z.clip(-4, 6), bins=80, density=True, color=BLUE, alpha=0.85, label="data")
        ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, label="normal")
        ax.set_title(f"{title} (skewness {z.skew():.1f})")
        ax.set_xlabel("standard deviations from the asset's mean")
        ax.set_yticks([])
    axes[0].legend(loc="upper right")
    save(fig, out / "distribution.png")

    # 3. memory: realized variance versus squared returns
    by = panel.groupby("symbol")
    acf_rv = np.mean([autocorrelation(g["y"]) for _, g in by], axis=0)
    acf_r2 = np.mean([autocorrelation(g["r"] ** 2) for _, g in by], axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    lags = np.arange(1, LAGS + 1)
    ax.plot(lags, acf_rv, color=BLUE, label="log realized variance (input of HAR, MEM)")
    ax.plot(lags, acf_r2, color=ORANGE, label="squared daily return (input of EWMA, ARCH, GARCH)")
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_title(f"{dataset}: autocorrelation, average across assets")
    ax.set_xlabel("lag in trading days")
    ax.legend(loc="upper right")
    save(fig, out / "memory.png")
    print(f"written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()
    for name in (DATASETS if args.dataset == "all" else [args.dataset]):
        run(name)
