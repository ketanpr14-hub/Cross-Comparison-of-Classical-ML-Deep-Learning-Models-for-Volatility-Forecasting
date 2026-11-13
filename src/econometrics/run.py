"""
Modelling and testing: run every model on one dataset and write a structured result folder.

Run from the project root:
    python -m src.econometrics.run --dataset stocks
    python -m src.econometrics.run --dataset forex --split-policy expanding --refit-every 252
    python -m src.econometrics.run --dataset stocks --quick          # 3 assets, to check the code

Each run creates results/econometrics/<dataset>/<run id>/ with
    run_config.json        every setting of the run: dataset, time window, splitting policy,
                           hyperparameters, masked days, sample sizes, package versions
    metrics.csv            one row per cutoff and model: QLIKE, log-MSE, DM vs benchmark, MCS
    dm_qlike.csv           Diebold-Mariano statistic and p-value for every pair of models
    per_asset_metrics.csv  QLIKE and log-MSE of each model on each asset
    params.csv             estimated parameters of every model / asset / fold
    forecasts.csv.gz       every forecast (not committed, see .gitignore)
and adds one line to results/econometrics/runs.csv.
"""
import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

from . import evaluation as ev
from .config import ALL_MODELS, CUTOFFS, DATASETS, RESULTS_DIR, ROOT, RunConfig
from .data import build_panel
from .models import make_folds, run_models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = RunConfig()
    p.add_argument("--dataset", choices=list(DATASETS), default=d.dataset)
    p.add_argument("--start", help="first date kept, YYYY-MM-DD")
    p.add_argument("--end", help="last date kept, YYYY-MM-DD")
    p.add_argument("--symbols", nargs="+", help="subset of assets")
    p.add_argument("--cutoffs", nargs="+", default=list(CUTOFFS), help="test start dates")
    p.add_argument("--split-policy", choices=["fixed", "expanding", "rolling"], default=d.split_policy)
    p.add_argument("--refit-every", type=int, default=d.refit_every)
    p.add_argument("--window", type=int, default=d.window)
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    p.add_argument("--ewma-lambda", type=float, default=d.ewma_lambda)
    p.add_argument("--arch-p", type=int, default=d.arch_p)
    p.add_argument("--garch-dist", choices=["normal", "t"], default=d.garch_dist)
    p.add_argument("--har-pooling", choices=["pooled", "per_asset"], default=d.har_pooling)
    p.add_argument("--benchmark", default=d.benchmark)
    p.add_argument("--mcs-size", type=float, default=d.mcs_size)
    p.add_argument("--mcs-reps", type=int, default=d.mcs_reps)
    p.add_argument("--mcs-block", type=int, default=d.mcs_block)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--n-jobs", type=int, default=d.n_jobs)
    p.add_argument("--quick", action="store_true", help="3 assets and 200 bootstrap draws")
    args = vars(p.parse_args())
    args["cutoffs"] = tuple(args["cutoffs"])
    cfg = RunConfig(**args)
    if cfg.benchmark not in cfg.models:
        p.error(f"the benchmark {cfg.benchmark} must be one of the models that are run")
    if cfg.quick:
        cfg.mcs_reps = 200
    return cfg


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def evaluate_cutoff(panel, cutoff, cfg):
    """Forecast, then score every model on the same asset-days."""
    folds = make_folds(panel["date"], cutoff, cfg)
    H, LOGF, params = run_models(panel, folds, cfg)

    in_test = (panel["date"] >= cutoff) & panel["target_rv"].notna()
    missing = H[in_test].isna().sum()
    keep = in_test & H.notna().all(axis=1) & LOGF.notna().all(axis=1)
    sample, H, LOGF = panel[keep], H[keep], LOGF[keep]

    q, mse_log = ev.loss_tables(sample, H, LOGF)
    daily = ev.daily_mean(q, sample["date"])
    dm = ev.dm_matrix(daily, cfg.dm_lags)
    mcs = ev.model_confidence_set(daily, cfg.mcs_size, cfg.mcs_reps, cfg.mcs_block, cfg.seed)
    by_asset = ev.dm_by_asset(q, sample, cfg.benchmark, cfg.dm_lags)

    metrics = pd.DataFrame({"n_forecasts": len(sample), "qlike": q.mean(), "mse_log_rv": mse_log.mean()})
    metrics["qlike_vs_benchmark_pct"] = 100 * (metrics["qlike"] / metrics.loc[cfg.benchmark, "qlike"] - 1)
    vs = dm[dm["model_b"] == cfg.benchmark].set_index("model_a")
    metrics["dm_stat_vs_benchmark"] = vs["dm_stat"]
    metrics["dm_pvalue_vs_benchmark"] = vs["p_value"]
    metrics = metrics.join(by_asset).join(mcs)
    metrics = metrics.sort_values("qlike").rename_axis("model").reset_index()
    metrics.insert(0, "qlike_rank", np.arange(1, len(metrics) + 1))
    metrics.insert(0, "cutoff", cutoff)

    per_asset = pd.concat({"qlike": q, "mse_log_rv": mse_log}, axis=1).groupby(sample["symbol"]).mean()
    per_asset = per_asset.stack(level=1, future_stack=True).rename_axis(["symbol", "model"]).reset_index()
    per_asset.insert(0, "cutoff", cutoff)

    forecasts = sample[["date", "symbol", "target_date", "target_rv"]].join(H)
    forecasts.insert(0, "cutoff", cutoff)
    info = {
        "cutoff": cutoff, "n_folds": len(folds),
        "train_start": str(folds[0].train_start.date()),
        "test_first_origin": str(sample["date"].min().date()),
        "test_last_origin": str(sample["date"].max().date()),
        "n_test_days": int(sample["date"].nunique()), "n_test_forecasts": int(len(sample)),
        "n_train_rows_before_cutoff": int((panel["date"] < cutoff).sum()),
        "test_rows_without_forecast_by_model": {m: int(n) for m, n in missing.items() if n},
    }
    dm.insert(0, "cutoff", cutoff)
    params.insert(0, "cutoff", cutoff)
    return metrics, dm, per_asset, params, forecasts, info


def main():
    cfg = parse_args()
    t0 = time.time()
    panel, mask_log = build_panel(cfg)
    print(f"{cfg.dataset}: {len(panel):,} rows, {panel['symbol'].nunique()} assets, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}, {len(mask_log)} masked days")

    outputs = []
    for cutoff in cfg.cutoffs:
        outputs.append(evaluate_cutoff(panel, cutoff, cfg))
        print(f"\ncutoff {cutoff}  ({time.time() - t0:.0f}s)")
        print(outputs[-1][0].drop(columns=["cutoff"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    metrics, dm, per_asset, params, forecasts = (pd.concat([o[i] for o in outputs], ignore_index=True)
                                                 for i in range(5))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{cfg.split_policy}" + ("_quick" if cfg.quick else "")
    out = RESULTS_DIR / cfg.dataset / run_id
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    dm.to_csv(out / "dm_qlike.csv", index=False)
    per_asset.to_csv(out / "per_asset_metrics.csv", index=False)
    params.to_csv(out / "params.csv", index=False)
    forecasts.to_csv(out / "forecasts.csv.gz", index=False)

    not_converged = params[~params["converged"].astype(bool)]
    run_config = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "settings": cfg.to_dict(),
        "data": {
            "file": DATASETS[cfg.dataset].relative_to(ROOT).as_posix(),
            "first_date": str(panel["date"].min().date()), "last_date": str(panel["date"].max().date()),
            "n_assets": int(panel["symbol"].nunique()), "assets": sorted(panel["symbol"].unique()),
            "n_rows": int(len(panel)),
            "target": "log(rv5) of the next session; QLIKE is scored on rv5 itself",
            "returns": "open-to-close log return",
            "masked_asset_days": [{"date": str(r.date.date()), "symbol": r.symbol, "reason": r.reason}
                                  for r in mask_log.itertuples()],
        },
        "splits": [o[5] for o in outputs],
        "testing": {
            "main_loss": "QLIKE = rv/h - log(rv/h) - 1 (lower is better)",
            "common_sample": "every model is scored on the same asset-days",
            "tests_on": "daily loss averaged across assets",
            "dm": "two-sided Diebold-Mariano, Newey-West variance, negative = better than benchmark",
            "mcs": "Hansen-Lunde-Nason, range statistic, stationary bootstrap",
        },
        "models_not_converged": not_converged[["cutoff", "model", "symbol", "fold"]].to_dict("records"),
        "environment": {"python": platform.python_version(), "git_commit": git_commit(),
                        **{pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "arch")}},
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # one line per run, to find a run again
    best = metrics.loc[metrics.groupby("cutoff")["qlike"].idxmin()]
    line = pd.DataFrame([{
        "run_id": run_id, "dataset": cfg.dataset, "first_date": run_config["data"]["first_date"],
        "last_date": run_config["data"]["last_date"], "n_assets": run_config["data"]["n_assets"],
        "split_policy": cfg.split_policy, "refit_every": cfg.refit_every, "window": cfg.window,
        "cutoffs": " ".join(cfg.cutoffs), "n_models": len(cfg.models), "har_pooling": cfg.har_pooling,
        "garch_dist": cfg.garch_dist, "quick": cfg.quick,
        "best_model": " | ".join(f"{r.cutoff}: {r.model} ({r.qlike:.4f})" for r in best.itertuples()),
        "folder": out.relative_to(ROOT).as_posix(),
    }])
    registry = RESULTS_DIR / "runs.csv"
    line.to_csv(registry, mode="a", header=not registry.exists(), index=False)

    if len(not_converged):
        print(f"\nwarning: {len(not_converged)} estimations did not converge (listed in run_config.json)")
    print(f"\nresults in {out.relative_to(ROOT)}   total time {time.time() - t0:.0f}s")
    print(f"display them with: python -m src.econometrics.report --run {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
