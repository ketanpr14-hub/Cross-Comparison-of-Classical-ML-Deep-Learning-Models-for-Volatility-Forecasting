"""
Find the rows behind non-finite or extreme losses.

    cd code
    python -m linear_ml.diagnose --dataset forex                 # fixed split
    python -m linear_ml.diagnose --dataset forex --split expanding

Prints:
  1. non-finite values in the features and targets;
  2. rows whose features are extreme compared with the rest of the data;
  3. per cutoff: sigma2 of each model (a huge sigma2 makes every variance forecast infinite),
     models whose QLIKE is not finite or very large, and the worst forecasts.
"""
import argparse

import numpy as np
import pandas as pd

from econometrics import evaluation as ev
from econometrics.config import CUTOFFS, RunConfig
from econometrics.models import make_folds

from .run import FEATURE_SETS, MODELS, build_specs, load, run_fold

FEATURES = sorted(set(sum(FEATURE_SETS.values(), [])))
COLS = FEATURES + ["target", "target_rv"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="forex")
    p.add_argument("--split", choices=["fixed", "expanding", "rolling"], default="fixed")
    args = p.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    cfg = RunConfig(dataset=args.dataset, split_policy=args.split)
    panel = load(args.dataset)
    specs = build_specs(MODELS, list(FEATURE_SETS))

    # 1. non-finite values
    nonfinite = (~np.isfinite(panel[COLS])).sum()
    print("\n1. Non-finite values per column:", "none" if nonfinite.sum() == 0 else "")
    if nonfinite.sum():
        print(nonfinite[nonfinite > 0].to_string())

    # 2. extreme rows: robust z-score above 15 on any feature
    x = panel[FEATURES]
    med, mad = x.median(), 1.4826 * (x - x.median()).abs().median()
    z = ((x - med) / mad.replace(0, np.nan)).abs()
    extreme = z.max(axis=1) > 15
    print(f"\n2. Rows with an extreme feature (robust z > 15): {int(extreme.sum())}")
    print(pd.DataFrame({"min": x.min(), "1%": x.quantile(.01), "median": med,
                        "99%": x.quantile(.99), "max": x.max()}).round(3).to_string())
    if extreme.any():
        worst = z.max(axis=1)[extreme].nlargest(15).index
        print(panel.loc[worst, ["date", "symbol"] + FEATURES].round(3).to_string())

    # 3. per cutoff
    for cutoff in CUTOFFS:
        parts = [run_fold(panel, f, k, specs, "pooled", cfg.min_train)
                 for k, f in enumerate(make_folds(panel["date"], cutoff, cfg))]
        test = pd.concat([pt[0] for pt in parts])
        logf = pd.concat([pt[1] for pt in parts])
        h = pd.concat([pt[2] for pt in parts])
        tuning = pd.DataFrame([r for pt in parts for r in pt[3] if r["selected"]])
        q = pd.DataFrame({m: ev.qlike(test["target_rv"].to_numpy(), h[m].to_numpy()) for m in h},
                         index=test.index)

        print(f"\n===== 3. cutoff {cutoff}, {args.split}")
        print("sigma2 of the selected candidate, per model and fold (largest first):")
        print(tuning.groupby("model")["sigma2"].max().sort_values(ascending=False).head(8).round(4).to_string())

        stats = pd.DataFrame({"nonfinite_forecasts": (~np.isfinite(h)).sum(),
                              "nonfinite_losses": (~np.isfinite(q)).sum(),
                              "max_loss": q.max()})
        bad = stats.index[(stats["nonfinite_losses"] > 0) | (stats["max_loss"] > 1e6)].tolist()
        print("\nmodels with non-finite or huge losses:", bad or "none")
        print(stats.sort_values("max_loss", ascending=False).head(8).to_string())

        for m in bad[:3]:
            worst = q[m].replace(np.nan, np.inf).nlargest(5).index
            print(f"\nworst rows for {m}:")
            print(test.loc[worst, ["date", "symbol", "target"] + FEATURES]
                  .assign(log_forecast=logf.loc[worst, m], variance_forecast=h.loc[worst, m])
                  .round(3).to_string())


if __name__ == "__main__":
    main()
