"""
Result display: tables and figures of one run.

    python -m src.econometrics.report                       # latest run of every dataset
    python -m src.econometrics.report --dataset stocks      # latest run of one dataset
    python -m src.econometrics.report --run results/econometrics/stocks/<run id>

Writes into the run folder
    report.md                 settings, ranking table per cutoff, reading guide
    figures/qlike_ranking.png QLIKE of every model, models of the Model Confidence Set highlighted
    figures/dm_matrix.png     Diebold-Mariano statistic for every pair of models
    figures/forecast_path.png best model and GARCH against realized volatility on one asset
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS, RESULTS_DIR, ROOT
from .plots import BLUE, DIVERGING, GREY, LIGHT_GREY, ORANGE, TEXT, TEXT_2, plt, save, use_style


def latest_run(dataset):
    runs = sorted(p for p in (RESULTS_DIR / dataset).glob("*") if (p / "metrics.csv").exists()
                  and not p.name.endswith("_quick"))
    return runs[-1] if runs else None


def ranking_figure(metrics, cfg, out):
    cutoffs = list(metrics["cutoff"].unique())
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(4.6 * len(cutoffs), 0.3 * metrics["model"].nunique() + 1.4),
                             squeeze=False)
    for ax, cutoff in zip(axes[0], cutoffs):
        m = metrics[metrics["cutoff"] == cutoff].sort_values("qlike", ascending=False)
        colors = [BLUE if inside else GREY for inside in m["in_mcs"]]
        ax.barh(m["model"], m["qlike"], color=colors, height=0.62)
        for y, value in enumerate(m["qlike"]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=8, color=TEXT)
        ax.set_xlim(0, m["qlike"].max() * 1.14)
        ax.set_title(f"Test from {cutoff}")
        ax.set_xlabel("QLIKE (lower is better)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GREY)]
    confidence = round(100 * (1 - cfg["settings"]["mcs_size"]))
    fig.legend(handles, [f"in the {confidence} % Model Confidence Set", "rejected"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    return save(fig, out / "figures" / "qlike_ranking.png")


def dm_figure(dm, metrics, out):
    cutoffs = list(dm["cutoff"].unique())
    n = metrics["model"].nunique()
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(0.42 * n * len(cutoffs) + 3.2, 0.36 * n + 1.6), squeeze=False)
    for ax, cutoff in zip(axes[0], cutoffs):
        order = metrics[metrics["cutoff"] == cutoff].sort_values("qlike")["model"].tolist()
        table = dm[dm["cutoff"] == cutoff].pivot(index="model_a", columns="model_b", values="dm_stat")
        table = table.reindex(index=order, columns=order)
        significant = table.where(table.abs() > 1.96, 0.0)       # not significant = neutral grey
        image = ax.imshow(significant.clip(-5, 5), cmap=DIVERGING, vmin=-5, vmax=5)
        ax.set_xticks(range(n), order, rotation=90)
        ax.set_yticks(range(n), order)
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"Test from {cutoff}")
    bar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.7, pad=0.02)
    bar.set_label("DM statistic: blue = row model better than column model,\n"
                  "orange = worse, grey = no significant difference at 5 %", fontsize=8)
    bar.outline.set_visible(False)
    return save(fig, out / "figures" / "dm_matrix.png")


def path_figure(run_dir, metrics, out, days=250):
    path = run_dir / "forecasts.csv.gz"
    if not path.exists():
        return None
    f = pd.read_csv(path, parse_dates=["date", "target_date"])
    cutoff = metrics["cutoff"].iloc[-1]
    best = metrics[metrics["cutoff"] == cutoff].sort_values("qlike")["model"].iloc[0]
    other = "GARCH" if "GARCH" in f.columns and best != "GARCH" else None
    f = f[f["cutoff"] == cutoff]
    symbol = f["symbol"].iloc[0]
    f = f[f["symbol"] == symbol].tail(days)

    def annualized(variance):
        return 100 * np.sqrt(252 * variance)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(f["target_date"], annualized(f["target_rv"]), color=GREY, linewidth=1, label="realized")
    if other:
        ax.plot(f["target_date"], annualized(f[other]), color=ORANGE, label=f"{other} forecast")
    ax.plot(f["target_date"], annualized(f[best]), color=BLUE, label=f"{best} forecast (best QLIKE)")
    ax.set_title(f"{symbol}: one-day-ahead volatility forecasts, last {len(f)} test days")
    ax.set_ylabel("annualized volatility, %")
    ax.legend(loc="upper left", ncol=3)
    return save(fig, out / "figures" / "forecast_path.png")


def report(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "metrics.csv", dtype={"cutoff": str})
    dm = pd.read_csv(run_dir / "dm_qlike.csv", dtype={"cutoff": str})
    s, d = cfg["settings"], cfg["data"]

    use_style()
    ranking_figure(metrics, cfg, run_dir)
    dm_figure(dm, metrics, run_dir)
    has_path = path_figure(run_dir, metrics, run_dir)

    policy = {"fixed": "fixed: one estimation on all data before the cutoff",
              "expanding": f"expanding: re-estimated every {s['refit_every']} days on all past data",
              "rolling": f"rolling: re-estimated every {s['refit_every']} days on the last {s['window']} days"}
    lines = [
        f"# Econometric volatility forecasts: {s['dataset']}", "",
        f"Run `{cfg['run_id']}` ({cfg['created']}, {cfg['runtime_seconds']} s, commit {cfg['environment']['git_commit']}).", "",
        "## Settings", "",
        f"- **Dataset**: {s['dataset']}, {d['n_assets']} assets, {d['first_date']} to {d['last_date']} ({d['n_rows']:,} asset-days, "
        f"{len(d['masked_asset_days'])} masked)",
        f"- **Target**: {d['target']}",
        f"- **Splitting policy**: {policy[s['split_policy']]}; test cutoffs {', '.join(s['cutoffs'])}",
        f"- **Hyperparameters**: EWMA lambda {s['ewma_lambda']}, ARCH order {s['arch_p']}, GARCH errors {s['garch_dist']}, "
        f"HAR estimation {s['har_pooling']} with lags 1/{s['har_lags'][0]}/{s['har_lags'][1]}",
        f"- **Tests**: benchmark {s['benchmark']}; Model Confidence Set at {round(100 * (1 - s['mcs_size']))} % "
        f"({s['mcs_reps']} stationary-bootstrap draws, mean block {s['mcs_block']} days)", "",
    ]
    columns = {"qlike_rank": "rank", "model": "model", "qlike": "QLIKE", "qlike_vs_benchmark_pct": "vs benchmark, %",
               "dm_stat_vs_benchmark": "DM stat", "dm_pvalue_vs_benchmark": "DM p-value",
               "assets_sig_better": "assets better", "assets_sig_worse": "assets worse",
               "mcs_pvalue": "MCS p-value", "in_mcs": "in MCS", "mse_log_rv": "MSE log RV"}
    for split in cfg["splits"]:
        cutoff = split["cutoff"]
        table = metrics[metrics["cutoff"] == cutoff][list(columns)].rename(columns=columns)
        for col in ("assets better", "assets worse"):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        table["in MCS"] = table["in MCS"].map({True: "yes", False: ""})
        lines += [f"## Test from {cutoff}", "",
                  f"{split['n_test_forecasts']:,} forecasts on {split['n_test_days']:,} days "
                  f"({split['test_first_origin']} to {split['test_last_origin']}), {split['n_folds']} estimation(s).", "",
                  table.to_markdown(index=False, floatfmt=".4f").replace("nan", "   "), ""]
        print(f"\n{s['dataset']} | test from {cutoff} | {s['split_policy']} split")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep=""))
    lines += [
        "## How to read the table", "",
        "- **QLIKE** is the loss to minimize (0 = perfect forecast). Models are ranked on it.",
        f"- **DM stat / p-value**: Diebold-Mariano test against {s['benchmark']} on the daily loss averaged across assets. "
        "A negative statistic means the model beats the benchmark; the difference is significant at 5 % when the p-value is below 0.05.",
        f"- **assets better / worse**: number of assets (out of {d['n_assets']}) where the model is significantly better / worse "
        "than the benchmark at 5 % when the test is run asset by asset.",
        "- **in MCS**: the model belongs to the Model Confidence Set, the smallest group of models that contains the best one "
        "with the stated confidence. Models outside it are significantly worse than the best.",
        "- **MSE log RV** is the criterion of the DL and foundation-model branches, on the same asset-days when the dataset is stocks.", "",
        "## Figures", "", "![QLIKE ranking](figures/qlike_ranking.png)", "", "![Diebold-Mariano matrix](figures/dm_matrix.png)", "",
    ]
    if has_path:
        lines += ["![Forecast path](figures/forecast_path.png)", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report written to {(run_dir / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder; default = latest run")
    parser.add_argument("--dataset", choices=list(DATASETS), help="latest run of this dataset")
    args = parser.parse_args()
    if args.run:
        report(args.run)
    else:
        for name in ([args.dataset] if args.dataset else DATASETS):
            run = latest_run(name)
            if run:
                report(run)
            else:
                print(f"no run found for {name}")
