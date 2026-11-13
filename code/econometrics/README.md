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
python -m econometrics.run --dataset stocks       # models + tests, about 30 s
python -m econometrics.report                     # tables and figures of the latest runs
```

Other settings of `run` (all are saved with the results; `--help` lists them):

```sh
python -m econometrics.run --dataset forex --split-policy expanding --refit-every 252
python -m econometrics.run --dataset futures --split-policy rolling --window 1500
python -m econometrics.run --dataset stocks --start 2018-01-01 --cutoffs 2023-01-01
python -m econometrics.run --dataset stocks --models RW GARCH logHAR --garch-dist t
python -m econometrics.run --dataset stocks --quick      # 3 assets, to check the code
```

## Files

| File | Step |
|---|---|
| `eda.py` | exploratory analysis: summary table and three figures per dataset |
| `data.py` | cleaning (masked days) and feature engineering |
| `models.py` | the models and the splitting policies |
| `evaluation.py` | QLIKE, Diebold-Mariano, Model Confidence Set |
| `run.py` | runs everything on one dataset and writes the result folder |
| `report.py` | result display: `report.md` and figures for a run |
| `config.py` | paths, cutoffs, masked days, default hyperparameters |

## Models

| Name | Uses | Description |
|---|---|---|
| `RW` | RV | tomorrow = today |
| `EWMA` | returns | RiskMetrics smoothing of squared returns, lambda = 0.94 |
| `ARCH` | returns | ARCH(5) |
| `GARCH` | returns | GARCH(1,1) |
| `GJR-GARCH` | returns | GARCH(1,1) with leverage |
| `EGARCH` | returns | EGARCH(1,1) with leverage |
| `EWMA-RV` | RV | smoothing of realized variance, lambda chosen by QLIKE on the training days |
| `HAR` | RV | Corsi (2009): daily, weekly and monthly averages, in levels |
| `HARQ` | RV | Bollerslev, Patton & Quaedvlieg (2016): HAR corrected for measurement error |
| `logHAR` | RV | HAR in logs, the benchmark of the tests |
| `logHAR-MJ` | RV | logHAR + market-wide level + jump share |
| `logSHAR` | RV | Patton & Sheppard (2015): the daily lag split into upside and downside variance |
| `MEM` | RV | multiplicative error model (Engle, 2002): a GARCH(1,1) recursion on RV itself |
| `AMEM` | RV | MEM where a negative-return day raises tomorrow's variance more |

Returns are open-to-close log returns. ARCH-type models and MEM are estimated asset by
asset; the HAR regressions pool all assets (`--har-pooling per_asset` changes that).
Parameters are estimated on the training part only and then kept fixed while the model is
rolled through the test days.

## Output of a run

`results/econometrics/<dataset>/<date-time>_<split policy>/`

| File | Content |
|---|---|
| `run_config.json` | dataset, time window, splitting policy, hyperparameters, masked days, sample sizes, non-converged fits, package versions, git commit |
| `metrics.csv` | per cutoff and model: QLIKE, rank, MSE on log RV (`mse_log_rv`) and on RV (`mse_rv`), DM test against the benchmark, number of assets significantly better / worse, MCS p-value and membership |
| `dm_qlike.csv` | DM statistic and p-value for every pair of models |
| `per_asset_metrics.csv` | QLIKE and both MSE of every model on every asset |
| `params.csv` | estimated parameters of every model, asset and fold |
| `report.md`, `figures/` | written by `report.py` |
| `forecasts.csv.gz` | every forecast; not versioned |

`results/econometrics/runs.csv` has one line per run. `results/econometrics/eda/` holds
the EDA output.

## Results (fixed split, the protocol of the other branches)

QLIKE of the best model, of the benchmark and of the best return-based model:

| Dataset | Test from | Best model | QLIKE | logHAR | GARCH | Models in the 90 % MCS |
|---|---|---|---|---|---|---|
| stocks | 2019 | logHAR-MJ | 0.1818 | 0.1863 | 0.2830 | logHAR-MJ |
| stocks | 2021 | logHAR-MJ | 0.1772 | 0.1813 | 0.2460 | logHAR-MJ |
| forex | 2019 | logSHAR | 0.1527 | 0.1534 | 0.2018 | logSHAR, logHAR-MJ, logHAR, MEM, AMEM |
| forex | 2021 | logSHAR | 0.1569 | 0.1576 | 0.1855 | logSHAR, logHAR-MJ, logHAR, MEM, AMEM |
| futures | 2019 | AMEM | 0.1683 | 0.1723 | 0.2080 | AMEM, MEM, logSHAR, logHAR, logHAR-MJ, HARQ |
| futures | 2021 | AMEM | 0.1685 | 0.1699 | 0.1936 | AMEM, logSHAR, MEM, logHAR, logHAR-MJ, HARQ |

- The log-HAR variants and the MEM, which see intraday realized variance, beat every
  return-based model on the three datasets. EWMA, ARCH and GARCH have a QLIKE 14 % to 92 % above logHAR, and the
  difference is significant in every case.
- On stocks, adding the market level and the jump share to logHAR lowers QLIKE by 2.2 %
  to 2.5 %, and it is the only model left in the Model Confidence Set.
- On forex and futures (5 assets each) the log-HAR variants and the MEM cannot be told
  apart: all stay in the Model Confidence Set.
- HAR in levels is clearly worse than HAR in logs.
- Re-estimating every year on an expanding window (stocks) gives the same ranking.

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
  about 60 times that of the other models because its forecast explodes on a few days of 2020.
- With 5 assets, the forex and futures tests have much less power than the stocks tests.
