"""
Extra Diebold-Mariano tests, computed from the forecasts saved by run_dl.py (no retraining).

    python src/dl/dm_tests.py                 # every forecasts_*.csv.gz in results/dl/

For each dataset, split and cutoff it tests every model against the two HAR baselines
(pooled and Jean's per-asset), on the two losses: QLIKE and squared error of log RV.
The test uses one loss per day (the average over the assets of that day).
Negative statistic = the model is better than the baseline; p < 0.05 = significant.

Output: results/dl/dm_tests.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))                  # Jean's code
from econometrics.evaluation import dm_test, qlike      # noqa: E402

BASELINES = ["logHAR-MJ (pooled)", "logHAR-MJ (Jean, per asset)"]
OUT = ROOT / "results" / "dl"

rows = []
for path in sorted(OUT.glob("forecasts_*.csv.gz")):
    if "_quick" in path.name:
        continue
    dataset, split = path.name.removeprefix("forecasts_").removesuffix(".csv.gz").split("_")
    fc = pd.read_csv(path, parse_dates=["date"])
    models = [c.removesuffix(" log") for c in fc.columns if c.endswith(" log")]
    for cutoff, g in fc.groupby("cutoff"):
        # one loss per day and per model
        daily = {"QLIKE": {}, "MSE log RV": {}}
        for m in models:
            daily["QLIKE"][m] = qlike(g["target_rv"], g[f"{m} var"]).groupby(g["date"]).mean()
            daily["MSE log RV"][m] = ((g["target"] - g[f"{m} log"]) ** 2).groupby(g["date"]).mean()
        for loss, by_model in daily.items():
            for base in BASELINES:
                for m in models:
                    if m == base:
                        continue
                    stat, p = dm_test(by_model[m], by_model[base])
                    rows.append({"dataset": dataset, "split": split, "cutoff": cutoff, "loss": loss,
                                 "model": m, "baseline": base,
                                 "model loss": by_model[m].mean(), "baseline loss": by_model[base].mean(),
                                 "DM stat": stat, "DM p-value": p})

table = pd.DataFrame(rows)
table.to_csv(OUT / "dm_tests.csv", index=False)
print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nsaved results/dl/dm_tests.csv ({len(table)} tests)")
