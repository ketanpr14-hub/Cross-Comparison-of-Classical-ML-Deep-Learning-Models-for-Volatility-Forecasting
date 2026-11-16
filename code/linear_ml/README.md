# Ridge, lasso and k-NN for one-day-ahead volatility

Do penalized linear models and nearest neighbors forecast tomorrow's realized variance better than the HAR regression? This part uses the project's shared protocol, so its numbers go in the same tables as the econometrics, tree, deep learning and foundation-model branches.

## Protocol

Same as the rest of the project. Data, cleaning, target, cutoffs, folds and losses are imported from `code/econometrics` (`data.py`, `models.py`, `evaluation.py`), not copied.

| | |
|---|---|
| Target | `log(rv5)` of the next session, per asset |
| Cutoffs | test from 2019-01-01 and from 2021-01-01 |
| Splits | `fixed` (main): train once before the cutoff. `expanding`: refit every 252 days on all past data. `rolling`: refit every 252 days on the last 1,000 days. Same folds as Jean's and Quentin's runs |
| Validation | the last **two** years before each fold's first test day, walk-forward as in `econometrics/tuning.py`: each candidate is fitted on everything before a validation year and forecasts it; the lowest QLIKE over both years wins. The same errors give sigma2. `--tune-years 1` keeps only the last year, as in the DL branch |
| Variance forecast | `h = exp(log forecast + sigma2 / 2)`, sigma2 = variance of the validation log errors, same rule for every model (as in the DL branch) |
| Main loss | QLIKE on `rv5`; also MSE on `log(rv5)` and on `rv5`. **Lower is better** |
| Tests | Diebold-Mariano against `logHAR-MJ` and 90 % Model Confidence Set, on the daily loss averaged across assets |
| Test rows | the rows of `build_panel` where every feature exists; identical for every model in a run |

Nothing after a fold's first test day is used to fit, tune or calibrate it.

## Models

Each model is run on three feature sets, to separate the effect of the *model* from the effect of the *information* it gets.

| Feature set | Columns of `build_panel` |
|---|---|
| `har` | `x_day`, `x_week`, `x_month`: log RV over the last 1, 5 and 22 days |
| `har_ext` | `har` + `x_pos`, `x_neg` (good and bad semivariance), `jump_share`, `log_rq` (realized quarticity) |
| `har_mkt` | `har_ext` + `m_day`, `m_week`: market-wide log RV of the asset class |

| Model | Description | Hyperparameters (chosen on the validation years) |
|---|---|---|
| `logHAR-MJ` | the group benchmark: OLS on `x_day`, `x_week`, `x_month`, `m_day`, `m_week`, `jump_share` | none |
| `RW` | tomorrow = today | none |
| `OLS` | linear regression on the feature set (HAR on `har`, HAR-X on the others) | none |
| `Ridge` | L2-penalized regression on standardized features | alpha in 10⁻² … 10⁵ (15 values) |
| `Lasso` | L1-penalized regression; sets some coefficients to exactly zero | alpha in 10⁻⁵ … 10⁻¹ (13 values) |
| `kNN-level` | average of tomorrow's log RV over the k most similar past days (Euclidean distance on standardized features) | k in 10, 25, 50, 100, 200; uniform or distance weights |
| `kNN-change` | today's log RV + the neighbors' average change | same as kNN-level |

The two grids differ because scikit-learn's ridge penalizes the *sum* of squared errors (so alpha grows with the number of rows) while lasso penalizes the *mean* (so alpha is in units of the target).

`kNN-level` can never forecast above the highest value in its training data, so it is capped during a volatility spike that training never saw (March 2020 for the 2019 cutoff). `kNN-change` adds the neighbors' change to today's level, so it can follow the spike. Comparing the two isolates this limitation, which the tree models and networks also share.

By default every model is pooled: one model for all assets of a dataset, as in the tree and DL branches. `--pooling per_asset` fits one model per asset instead (the hyperparameters are still chosen for all assets together).

## Run it

Python 3.12, VOLARE folders under `data/` at the project root (not versioned). From the `code/` folder:

```sh
cd code
python -m pip install -r linear_ml/requirements.txt

python -m linear_ml.run --dataset stocks --quick              # 3 assets: checks the setup, numbers mean nothing
python -m linear_ml.run --dataset stocks                      # fixed split
python -m linear_ml.run --dataset stocks --split expanding
python -m linear_ml.run --dataset stocks --split rolling
python -m linear_ml.run --dataset futures --pooling per_asset
python -m linear_ml.run --dataset combined                    # the three files as one panel
python -m linear_ml.run --dataset forex --models Lasso kNN-change --feature-sets har har_mkt
python -m linear_ml.run --dataset forex --tune-years 1        # one validation year (robustness)
```

## Output (`results/linear_ml/<dataset>/<split>[_per_asset][_tune1]/`)

| File | Content |
|---|---|
| `metrics.csv` | per cutoff and model: `n_forecasts`, QLIKE, MSE on log RV and on RV, % difference from `logHAR-MJ`, DM statistic and p-value against it, MCS p-value and membership |
| `tuning.csv` | every hyperparameter candidate per fold: validation QLIKE, sigma2, whether it was selected |
| `coefficients.csv` | standardized coefficients of ridge and lasso per fold: which features lasso keeps |
| `forecasts.csv.gz` | every log forecast; not versioned |

In `vs_benchmark_pct`, negative means better than `logHAR-MJ`. A negative DM statistic with a p-value under 0.05 means significantly better on QLIKE.

## Why two validation years

With one validation year, the 2019 cutoff tunes on 2018, a calm year for FX after a much more volatile training period. Heavily shrunk, nearly constant forecasts scored best on that one year (QLIKE penalizes over-prediction less than under-prediction), so lasso picked the largest penalty in its grid and ridge one of the largest. Those forecasts then failed from 2019 onward: ridge and lasso ended 5–23 % worse than `logHAR-MJ` on FX, while OLS on the same features was the best model. The problem is visible in the validation data alone (the selected penalty sits at the edge of the grid, driven by a single year), and two validation years is the rule of `econometrics/tuning.py`, the closest analogue in the project. The change applies to every dataset and split; the one-year runs are kept as a robustness check (`--tune-years 1`, folders ending in `_tune1`).

## Results

To be filled in from `results/linear_ml/` after the full runs.

## Earlier version

The first version of this work forecast 5-day volatility with its own data pipeline and a rolling 1,232-day window refit monthly, with tests from 2020. It is kept on the branch `ketan-5day-horizon` as a possible extension (does the ranking change at a weekly horizon?), but its numbers cannot be compared with the rest of the project.
