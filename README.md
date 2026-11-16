# Cross-Comparison-of-Classical-ML-Deep-Learning-Models-for-Volatility-Forecasting
# Volatility Forecasting: Ridge, Lasso and k-NN

This branch contains the data pipeline and the linear and nearest-neighbor models for our group project comparing classical, machine learning and deep learning models for volatility forecasting across asset classes (FX, futures, stocks).

**Question:** do ridge, lasso and k-NN forecast next-week volatility better than a naive benchmark and the classic HAR model, and does extra information (other realized measures, market-wide volatility) help?

## What's in this branch

```
config/
    config.py                 shared settings: paths, dates, walk-forward sizes, feature sets
    build_mkt_features.py     adds market-average and other-symbol volatility features
data_cleaning_preprocessing/
    load_data.py              splits raw files into one series per symbol
    process_data.py           builds features (X) and the target (y)
    walk_forward.py           rolling walk-forward train/test splits
Model Training & Evaluation/
    naive_baseline_and_ols.py naive benchmark and OLS (HAR / HAR-X)
    ridge_lasso.py            ridge and lasso
    knearest_neighbors.py     k-NN, two versions
    compute_results.py        results computation
    plot_results.py           charts and Excel tables comparing all models
results/
    figures/                  interactive HTML charts
    tables/                   results.xlsx (summary and by-symbol losses)
```

## Data

The dataset is **not included** in this repository. It is a realized-volatility dataset with daily measures computed from intraday prices (e.g. `rv5`, 5-minute realized variance) for FX pairs, futures and stocks. To run the code, place the raw files in `Data/` and run the pipeline below.

## How to run

Run each step from the repository root, in this order:

1. `data_cleaning_preprocessing/load_data.py`: one cleaned file per symbol
2. `data_cleaning_preprocessing/process_data.py`: features and target
3. `config/build_mkt_features.py`: market and other-symbol features
4. `Model Training & Evaluation/naive_baseline_and_ols.py`: naive and OLS forecasts (needed by the other scripts' comparisons)
5. `Model Training & Evaluation/ridge_lasso.py` and `knearest_neighbors.py`
6. `Model Training & Evaluation/plot_results.py`: charts and Excel tables

All settings live in `config/config.py`. Change them there, not in individual scripts, so every model uses the same setup.

## Setup

### Target

Log realized volatility over the **next 5 trading days**, built from 5-minute realized variance:

y_t = log( sqrt( rv5[t+1] + ... + rv5[t+5] ) )

All features use information up to and including day t only.

### Feature sets

Every model is run on each of three feature sets, so we can separate the effect of the *model* from the effect of the *information* it gets.

| Set | Features |
| --- | --- |
| `har` | Log of `rv5` averaged over the last 1, 5 and 22 days (the HAR terms) |
| `har_ext` | `har` + positive and negative semivariances, jump share, Garman-Klass range estimator |
| `har_mkt` | `har_ext` + leave-one-out market averages of the HAR terms (other symbols in the same asset class) + each other symbol's daily volatility |

### Walk-forward evaluation

| Setting | Value |
| --- | --- |
| Out-of-sample period | 2020-01-01 to end of data, identical for every model and symbol |
| Training window | Rolling, 1,250 observations (about 5 years) |
| Refit frequency | Every 21 observations (about 1 month) |
| Gap | 5 observations between training and test, so no training target overlaps the test period |
| Hyperparameter tuning | Time-series cross-validation (5 folds, gap of 5) inside each training window only |

Scaling is fitted inside each training window as part of a scikit-learn `Pipeline`, so no information from the test period is used.

## Models

**Naive:** next week's volatility equals last week's. Every result is reported as a ratio to this benchmark.

**OLS (HAR / HAR-X):** linear regression on each feature set. On `har` this is the standard HAR model; on the larger sets it is the linear benchmark with the same information as the ML models.

**Ridge and lasso:** penalized linear regression on standardized features. The penalty is chosen per fold from a log-spaced grid (ridge: 10⁻³ to 10³; lasso: 10⁻⁵ to 10⁻¹; the two are scaled differently in scikit-learn). Lasso also selects features, which shows which realized measures matter for each asset class.

**k-NN:** forecasts from the most similar past days, using Euclidean distance on standardized features. k and the weighting are tuned per fold. Two versions:

- `knn_level` averages the neighbors' future volatility. It cannot forecast above the highest value in its training window.
- `knn_change` forecasts the current level plus the neighbors' average *change* in volatility, so it can extrapolate during spikes such as March 2020. Comparing the two shows how much this limitation matters in high-volatility periods.

## Evaluation

- **Losses:** QLIKE (the standard loss for volatility forecasts, computed on variances) and MSE on log volatility
- **Benchmark:** each model's loss divided by naive's loss on the same dates. Below 1 means the model beats naive.
- **Outputs:** `results/figures/` (median ratio by asset class, and a heatmap by symbol) and `results/tables/results.xlsx`

## Notes

- Converting log forecasts to variances (`exp(2 * forecast)`) slightly underestimates variance on average. This affects all models equally, so comparisons remain fair.
- Forecast files in `results/forecasts/` are not committed, because they contain the target, which is derived from the dataset.
