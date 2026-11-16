"""
Figure for the data-cleaning slide: one futures contract's realized volatility, with the days
the shared cleaning masks (contract rolls) marked.

    cd code
    python -m linear_ml.plot_masked_days                      # corn, the contract with most rolls
    python -m linear_ml.plot_masked_days --symbol NG          # natural gas

Writes results/linear_ml/figures/masked_days_<symbol>.png. Uses the same loading and cleaning
as every branch (econometrics.data), so the marked days are exactly the masked ones.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from econometrics.config import ROOT
from econometrics.data import clean, load_raw

NAMES = {"C": "Corn", "NG": "Natural gas", "GC": "Gold", "CL": "Crude oil", "ES": "S&P 500 E-mini"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
LINE, ACCENT = "#2a78d6", "#e34948"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="futures")
    p.add_argument("--symbol", default="C")
    args = p.parse_args()

    raw = load_raw(args.dataset)
    _, masked = clean(raw, args.dataset)
    s = raw[raw["symbol"] == args.symbol].set_index("date")
    m = masked[masked["symbol"] == args.symbol]
    vol = np.sqrt(252 * s["rv5"])                    # annualized volatility, easier to read than variance
    mvol = np.sqrt(252 * m.set_index("date")["rv5"])

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    ax.plot(vol.index, vol, color=LINE, lw=0.8, label="Daily realized volatility (from 5-minute returns)")
    ax.scatter(mvol.index, mvol, s=46, facecolors="none", edgecolors=ACCENT, linewidths=1.6, zorder=3,
               label=f"Masked contract-roll days ({len(m)})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Annualized volatility (log scale)", color=MUTED)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    name = NAMES.get(args.symbol, args.symbol)
    ax.set_title(f"{name} futures: contract rolls create artificial volatility spikes",
                 loc="left", fontsize=13, color=INK, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()

    out = ROOT / "results" / "linear_ml" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"masked_days_{args.symbol}.png"
    fig.savefig(path, facecolor="white")
    print(f"{name}: {len(s):,} days, {len(m)} masked -> {path}")


if __name__ == "__main__":
    main()