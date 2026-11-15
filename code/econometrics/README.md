# Econometric models for one-day-ahead volatility

This part of the project measures how well classical and advanced econometric models
forecast tomorrow's realized variance on the three VOLARE files (40 stocks, 5 currency
pairs, 5 futures). It follows the protocol of the DL and foundation-model branches, so
the numbers can be put in the same table.

## Protocol

| | |
|---|---|
| Target | `log(rv5)` of the next session, asset by asset |
| Split | train before the cutoff, test after; cutoffs 2019-01-01 and 2021-01-01 |
| Bad days | the 6 stock-days of Quentin's EDA are masked |
| Main loss | QLIKE = `rv/h - log(rv/h) - 1`, where `h` is the variance forecast. It is a loss: **lower is better** |
| Also reported | MSE on `log(rv5)`, the criterion of the other branches, and MSE on `rv5` itself |
| Model comparison | Diebold-Mariano tests and the Model Confidence Set, on the daily loss averaged across assets |

On stocks the test sample is the one of `src/dl/stocks_har_mlp.py` (78,605 and 58,405
forecasts) and `logHAR`, `logHAR-MJ` and `RW` reproduce its "HAR base", "HAR base +
market + jump" and "tomorrow = today" MSE to four decimals.

## Run it

Python 3.12. Put the three VOLARE folders under `data/` at the project root (they are not
versioned), then from the `code/` folder:

```sh
cd code
python -m pip install -r econometrics/requirements.txt

python -m econometrics.eda                        # EDA of the three files
python -m econometrics.data --dataset futures     # what the cleaning step removes
python -m econometrics.run --dataset stocks       # tuning + models + tests, about 3 min
python -m econometrics.report                     # tables and figures of the latest runs
```

Other settings of `run` (all are saved with the results; `--help` lists them):

```sh
python -m econometrics.run --dataset forex --split-policy expanding --refit-every 252
python -m econometrics.run --dataset futures --split-policy rolling --window 1500
python -m econometrics.run --dataset stocks --start 2018-01-01 --cutoffs 2023-01-01
python -m econometrics.run --dataset combined      # the three files as one panel of 50 assets
python -m econometrics.run --dataset stocks --no-tune     # default hyperparameters, about 30 s
python -m econometrics.run --dataset stocks --no-tune --models RW GARCH logHAR --garch-dist t
python -m econometrics.run --dataset stocks --tune-years 1   # one validation year, as in the DL branch
python -m econometrics.run --dataset stocks --quick      # 3 assets, to check the code
```

## Files

| File | Step |
|---|---|
| `eda.py` | exploratory analysis: summary table and three figures per dataset |
| `data.py` | cleaning (masked days) and feature engineering |
| `models.py` | the models and the splitting policies |
| `evaluation.py` | QLIKE, Diebold-Mariano, Model Confidence Set |
| `tuning.py` | hyperparameter selection on validation years |
| `run.py` | runs everything on one dataset and writes the result folder |
| `report.py` | result display: `report.md` and figures for a run |
| `config.py` | paths, cutoffs, masked days, default hyperparameters |

## Models

| Name | Uses | Description |
|---|---|---|
| `RW` | RV | tomorrow = today |
| `EWMA` | returns | RiskMetrics smoothing of squared returns (default lambda = 0.94) |
| `ARCH` | returns | ARCH(p) (default p = 5) |
| `GARCH` | returns | GARCH(p,q) (default 1,1) |
| `GJR-GARCH` | returns | GARCH(1,q) with leverage |
| `EGARCH` | returns | EGARCH(1,q) with leverage |
| `EWMA-RV` | RV | smoothing of realized variance, lambda chosen by QLIKE on the training days |
| `HAR` | RV | Corsi (2009): daily, weekly and monthly averages, in levels |
| `HARQ` | RV | Bollerslev, Patton & Quaedvlieg (2016): HAR corrected for measurement error |
| `logHAR` | RV | HAR in logs, the benchmark of the tests |
| `logHAR-MJ` | RV | logHAR + market-wide level + jump share |
| `logSHAR` | RV | Patton & Sheppard (2015): the daily lag split into upside and downside variance |
| `MEM` | RV | multiplicative error model (Engle, 2002): a GARCH(p,q) recursion on RV itself |
| `AMEM` | RV | MEM where a negative-return day raises tomorrow's variance more |

Returns are open-to-close log returns. ARCH-type models and MEM are estimated asset by
asset; the HAR regressions are estimated either on all assets together or asset by asset.
Parameters are estimated on the training part only and then kept fixed while the model is
rolled through the test days.

## Hyperparameter tuning

The DL branch keeps the last year before the cutoff as a validation set and stops training
when the validation loss stops improving. These models have no training epochs, so there
is nothing to stop early; the same validation idea is used to choose their hyperparameters
(`tuning.py`):

1. the two years before the cutoff are two validation blocks of one year each. This is a
   walk-forward validation: for each block, every candidate is estimated on all the data
   before the block (the estimation window grows by one year from block 1 to block 2,
   it does not have a fixed length) and forecasts the block;
2. the candidate with the lowest QLIKE over both blocks, all assets together, is kept;
3. the model is re-estimated with it on the whole training period and then tested.

Nothing after the cutoff is used. One set of values is chosen per model and per cutoff.
The grids are small, so every candidate is tried (no search library is needed):

| Model | Candidates |
|---|---|
| `EWMA` | lambda in 0.80, 0.85, 0.90, 0.94, 0.97, 0.99 |
| `ARCH` | p in 1, 3, 5, 10, 22; normal or Student-t errors |
| `GARCH` | p and q in 1, 2; normal or Student-t errors |
| `GJR-GARCH`, `EGARCH` | q in 1, 2; normal or Student-t errors |
| `MEM`, `AMEM` | p and q in 1, 2 |
| `HAR`, `HARQ` | pooled or per asset; OLS or weighted least squares (weight 1 / RV) |
| `logHAR`, `logHAR-MJ`, `logSHAR` | pooled or per asset |

`RW` has nothing to tune and `EWMA-RV` already fits its decay on the training days.
`tuning.csv` lists every candidate with its validation QLIKE, and `report.md` shows the
selected values next to the validation QLIKE of the default values. `--no-tune` skips
this step and uses the defaults.

## Output of a run

`results/econometrics/<dataset>/<date-time>_<split policy>_<tuned or default>/`
(the runs made before tuning existed have no suffix: they used the default values)

| File | Content |
|---|---|
| `run_config.json` | dataset, time window, splitting policy, hyperparameters, masked days, sample sizes, non-converged fits, package versions, git commit |
| `metrics.csv` | per cutoff and model: QLIKE, rank, MSE on log RV (`mse_log_rv`) and on RV (`mse_rv`), DM test against the benchmark, number of assets significantly better / worse, MCS p-value and membership |
| `dm_qlike.csv` | DM statistic and p-value for every pair of models |
| `per_asset_metrics.csv` | QLIKE and both MSE of every model on every asset |
| `params.csv` | estimated parameters of every model, asset and fold |
| `tuning.csv` | every hyperparameter candidate, its validation QLIKE, whether it was selected |
| `per_class_metrics.csv` | `combined` only: QLIKE and both MSE of every model on stocks, forex and futures |
| `report.md`, `figures/` | written by `report.py` |
| `forecasts.csv.gz` | every forecast; not versioned |

`results/econometrics/runs.csv` has one line per run. `results/econometrics/eda/` holds
the EDA output.

## Results (fixed split, tuned hyperparameters)

QLIKE of the best model, of the benchmark and of GARCH:

| Dataset | Test from | Best model | QLIKE | logHAR | GARCH | Models in the 90 % MCS |
|---|---|---|---|---|---|---|
| stocks | 2019 | logHAR-MJ | 0.1811 | 0.1863 | 0.2724 | logHAR-MJ |
| stocks | 2021 | logHAR-MJ | 0.1766 | 0.1813 | 0.2431 | logHAR-MJ |
| forex | 2019 | logHAR-MJ | 0.1511 | 0.1534 | 0.2021 | logHAR-MJ, logSHAR, logHAR, AMEM, MEM, HARQ |
| forex | 2021 | logHAR-MJ | 0.1555 | 0.1576 | 0.1847 | logHAR-MJ, HAR, logSHAR, logHAR, MEM, AMEM |
| futures | 2019 | AMEM | 0.1674 | 0.1698 | 0.2115 | AMEM, logSHAR, logHAR-MJ, HARQ |
| futures | 2021 | logSHAR | 0.1661 | 0.1680 | 0.1928 | logSHAR, AMEM, logHAR-MJ, HAR, MEM, HARQ |

- The log-HAR variants and the MEM, which see intraday realized variance, beat every
  return-based model on the three datasets. EWMA, ARCH and GARCH have a QLIKE 15 % to 64 %
  above logHAR, and the difference is significant in every case.
- On stocks, adding the market level and the jump share to logHAR lowers QLIKE by 2.5 %
  to 2.8 %, and it is the only model left in the Model Confidence Set.
- On forex and futures (5 assets each) the log-HAR variants and the MEM cannot be told
  apart: most stay in the Model Confidence Set.
- Re-estimating every year on an expanding window (stocks) gives the same first three
  models and the same Model Confidence Set; QLIKE moves by less than 1 % for the log-HAR
  variants and falls by 2 % to 7 % for the ARCH-type models.

### What tuning changes

Test QLIKE with tuned hyperparameters against the default ones (the reference runs of
2026-11-13, kept in `results/`):

| Model | Selected most often | Change in test QLIKE |
|---|---|---|
| `HAR` | weighted least squares | -2 % to -14 % |
| `ARCH` | p = 10 or 22 instead of 5 | -9 % to -15 % |
| `HARQ` | weighted least squares | -11 % to +1 % |
| `GARCH`, `GJR-GARCH`, `EGARCH` | q = 2, Student-t errors on stocks | -6 % to +2 % |
| `logHAR-MJ` | one regression per asset | -0.3 % to -2 % |
| `MEM`, `AMEM` | p or q = 2 | -0.2 % to -1.5 % |
| `logHAR`, `logSHAR` | pooled (the default) on stocks and forex, per asset on futures | 0 % to -2 % |
| `EWMA` | lambda = 0.90 or 0.97 instead of 0.94 | 0 % to +11 % |

- The ranking of the model families does not change. Tuning mostly helps the weak
  models: HAR in levels estimated by weighted least squares ends between 1 % below and 4 % above
  logHAR, against 6 % to 18 % above before.
- The best models gain little (logHAR-MJ: 0.1818 to 0.1811 on stocks from 2019), because
  they have almost nothing to tune.
- EWMA gets worse on the test set in 5 cases out of 6: the lambda that was best on the
  two validation years is not the best afterwards. The gain seen on validation is not
  guaranteed out of sample.
- On stocks before 2019, every EGARCH candidate has a validation QLIKE above 100 (its
  forecast explodes on a few days), so its selection there is between bad options.

## The three datasets as one panel

`--dataset combined` stacks the three files: 50 assets, 2009-09-25 to 2026-10-30. Each
file is cleaned with its own rules, and the market level of an asset is still the average
of its own asset class. What is shared is the pooled regressions and the hyperparameters:
one set per model for the 50 assets. Stocks start in 2015, so before that date the panel
holds the 10 forex and futures assets only.

| Test from | Forecasts | Best model | QLIKE | logHAR | GARCH | Models in the 90 % MCS |
|---|---|---|---|---|---|---|
| 2019 | 98,543 | logHAR-MJ | 0.1769 | 0.1815 | 0.2591 | logHAR-MJ |
| 2021 | 73,259 | logHAR-MJ | 0.1736 | 0.1783 | 0.2327 | logHAR-MJ |

The test asset-days of each class are exactly those of the separate runs, so the QLIKE of
a class can be compared between the two (`per_class_metrics.csv`, and a table in
`report.md`):

| Class | Test from | Best model, separate run | QLIKE | Best model, combined run | QLIKE |
|---|---|---|---|---|---|
| stocks | 2019 | logHAR-MJ | 0.1811 | logHAR-MJ | 0.1811 |
| stocks | 2021 | logHAR-MJ | 0.1766 | logHAR-MJ | 0.1766 |
| forex | 2019 | logHAR-MJ | 0.1511 | logSHAR | 0.1509 |
| forex | 2021 | logHAR-MJ | 0.1555 | logHAR-MJ | 0.1555 |
| futures | 2019 | AMEM | 0.1674 | AMEM | 0.1674 |
| futures | 2021 | logSHAR | 0.1661 | AMEM | 0.1671 |

- The overall ranking is the one of stocks, which are 80 % of the forecasts. logHAR-MJ is
  the only model in the Model Confidence Set at both cutoffs.
- Stacking the datasets does not improve the forecasts. logHAR-MJ, MEM, AMEM, EWMA-RV and
  RW give the same QLIKE as in the separate runs (the difference is below 0.1 %): they are
  estimated asset by asset, and for logHAR-MJ the validation years select one regression
  per asset in both cases.
- logHAR and logSHAR move by -1 % to +2.5 % on forex and futures and by less than 0.5 %
  on stocks.
- One regression in levels for the three asset classes is harmful: when the pooled HAR
  or HARQ is selected, its QLIKE rises by 18 % to 119 % on forex and by 48 % on futures
  (2019). Variance levels differ too much between classes; the log models do not have
  this problem.
- Re-estimating every year on an expanding window keeps the same first eight models and
  the same Model Confidence Set. QLIKE falls by 0.5 % to 1 % for logHAR-MJ and by 2 % to
  6 % for the GARCH-type models and ARCH.
- Hyperparameters chosen for 50 assets are mostly those of stocks. This helps ARCH on
  forex and futures (-4 % to -7 % in three cases out of four, order 22 instead of 10)
  and EWMA in two cases (-4 % and -5 %), and changes the GARCH-type models by less than 2 %.

## What the cleaning does, and what to keep in mind

- **Stock prices are not split-adjusted**: the squared close-to-close return is on average
  2.7 times `rv5` (up to 13.6 times for one stock). This is why the ARCH-type models use
  open-to-close returns.
- **Futures**: 36 asset-days are masked. One is a corrupt gold observation (2009-12-29,
  daily variance of 0.96). The other 35 are days where more than 90 % of the variance comes
  from a single jump (`rv5 / bv5 > 10`), which in this file are almost all contract rolls
  in corn and natural gas. A few may be genuine news jumps (for instance corn on 2012-01-12).
  The list is in `results/econometrics/eda/futures/masked_days.csv`.
- **Forex**: nothing is masked.
- Return-based models forecast the variance of the open-to-close return, which is about
  8 % below `rv5` on average for stocks and forex. Their forecasts are not rescaled.
- HARQ stays in the futures Model Confidence Set despite a poor QLIKE: its loss comes
  from a few very large errors, so the test cannot reject it. It is not a good model there.
- The MSE on `rv5` is dominated by a few crisis days. On stocks from 2019, EGARCH has an MSE
  about 2.6 times that of logHAR (60 times with the default hyperparameters) because its
  forecast explodes on a few days of 2020.
- With 5 assets, the forex and futures tests have much less power than the stocks tests.
