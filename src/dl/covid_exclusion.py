"""
How much of the LSTM result with the test from 2019 comes from the COVID crash?

Same forecasts as run_dl.py (no retraining). We only drop the crash days from the scoring
and compare every model with pooled logHAR-MJ again.

    python src/dl/covid_exclusion.py          # every forecasts_*.csv.gz in results/dl/

Three samples, all with the test from 2019-01-01:
    all days              the numbers of the main tables
    without COVID crash   drop the days whose target falls between 2020-02-20 and 2020-04-30
                          (S&P 500 peak on 19 Feb 2020, NBER recession Feb-Apr 2020)
    without 2020          drop the whole year (robustness check)

For each model: MSE of log RV and QLIKE, the change against pooled HAR in %, and the
Diebold-Mariano test on the daily average loss (negative = better than HAR).

Output: results/dl/covid_exclusion.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))                  # Jean's code
from econometrics.evaluation import dm_test, qlike      # noqa: E402

HAR = "logHAR-MJ (pooled)"
CUTOFF = "2019-01-01"
CRASH_START, CRASH_END = pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")
OUT = ROOT / "results" / "dl"

rows = []
for path in sorted(OUT.glob("forecasts_*.csv.gz")):
    if "_quick" in path.name:
        continue
    dataset, split = path.name.removeprefix("forecasts_").removesuffix(".csv.gz").split("_")
    fc = pd.read_csv(path, parse_dates=["date"])
    fc = fc[fc["cutoff"] == CUTOFF].copy()

    # `date` is the day the forecast is made; the target is the next trading day.
    days = pd.Series(sorted(fc["date"].unique()))
    next_day = dict(zip(days, days.shift(-1)))
    fc["target_date"] = fc["date"].map(next_day)

    in_crash = fc["target_date"].between(CRASH_START, CRASH_END)
    in_2020 = fc["target_date"].dt.year == 2020
    samples = {"all days": fc,
               "without COVID crash": fc[~in_crash],
               "without 2020": fc[~in_2020]}

    models = [c.removesuffix(" log") for c in fc.columns if c.endswith(" log")]
    for sample_name, g in samples.items():
        # one loss per day and per model (average over the assets of that day)
        mse = {m: ((g["target"] - g[f"{m} log"]) ** 2).groupby(g["date"]).mean() for m in models}
        ql = {m: qlike(g["target_rv"], g[f"{m} var"]).groupby(g["date"]).mean() for m in models}
        for m in models:
            stat, p = dm_test(mse[m], mse[HAR]) if m != HAR else (float("nan"), float("nan"))
            # MSE over all rows (same as the main tables), not the mean of the daily means
            se = (g["target"] - g[f"{m} log"]) ** 2
            se_har = (g["target"] - g[f"{HAR} log"]) ** 2
            q, q_har = qlike(g["target_rv"], g[f"{m} var"]), qlike(g["target_rv"], g[f"{HAR} var"])
            rows.append({"dataset": dataset, "split": split, "sample": sample_name, "model": m,
                         "n_forecasts": len(g), "n_days": g["date"].nunique(),
                         "MSE log RV": se.mean(), "MSE vs HAR (%)": 100 * (se.mean() / se_har.mean() - 1),
                         "DM stat (MSE)": stat, "DM p-value (MSE)": p,
                         "QLIKE": q.mean(), "QLIKE vs HAR (%)": 100 * (q.mean() / q_har.mean() - 1)})

table = pd.DataFrame(rows)
table.to_csv(OUT / "covid_exclusion.csv", index=False)

# short view: change in MSE against pooled HAR, in %, and the DM p-value in brackets
table["view"] = table.apply(lambda r: f"{r['MSE vs HAR (%)']:+.1f}% ({r['DM p-value (MSE)']:.2g})", axis=1)
show = table[table["model"] != HAR]
for (dataset, split), g in show.groupby(["dataset", "split"], sort=False):
    print(f"\n{dataset} | {split} | test from {CUTOFF}  -- MSE log RV vs pooled HAR (DM p-value)")
    print(g.pivot(index="model", columns="sample", values="view")[list(samples)].to_string())
print(f"\nsaved results/dl/covid_exclusion.csv ({len(table)} rows)")
