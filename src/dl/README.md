# Deep learning models for one-day-ahead volatility (MLP, LSTM)

This part of the project tests whether a small neural network forecasts tomorrow's realized
variance better than the HAR regression of the econometrics branch. It uses Jean's
`build_panel` (in `code/econometrics/`), so the cleaning, the features and the test rows are
the same as his: 78,605 forecasts from 2019 and 58,405 from 2021 on the 40 stocks; 10,160 and 7,565 on the 5
currency pairs; 9,778 and 7,289 on the 5 futures.

## Protocol

| | |
|---|---|
| Target | `log(rv5)` of the next session, asset by asset |
| Cutoffs | test from 2019-01-01 and from 2021-01-01 |
| Splits | `fixed`: train once before the cutoff. `expanding`: refit every 252 days on all past data (Jean's folds). `rolling`: refit every 252 days on the last 1,000 days only |
| Validation | the last year before each test fold; used for early stopping and for sigma2 |
| Variance forecast | `h = exp(log forecast + sigma2 / 2)`, sigma2 = variance of the validation errors, same rule for every model |
| Losses | QLIKE (main), MSE on `log(rv5)`, MSE on `rv5`. **Lower is better** |
| Test | Diebold-Mariano on the daily average QLIKE, each model against pooled logHAR-MJ |

## Models

| Name | Description |
|---|---|
| `RW` | tomorrow = today |
| `logHAR-MJ (pooled)` | one linear regression for all stocks on the 6 features (day, week, month, market day, market week, jump share). The reference of the tables |
| `logHAR-MJ (Jean, per asset)` | the same regression, one per stock, from `econometrics.models` |
| `MLP [16,16]` | the same 6 features through two hidden layers of 16 units |
| `LSTM 22d`, `LSTM 66d` | reads the last 22 or 66 days of (log RV, market log RV, jump share), 32 hidden units |

Networks: Adam, learning rate 1e-3, weight decay 1e-4, batch 512, at most 30 epochs, early
stopping with patience 5, average of 3 seeds. Nothing was tuned on the test years.

## Run it

Python 3.12, VOLARE folders (stocks, forex, futures) under `data/` at the project root (not versioned).

```sh
pip install torch scikit-learn arch pandas numpy pyarrow scipy

python src/dl/run_dl.py --quick               # 3 stocks, 10 s: checks the setup, numbers mean nothing
python src/dl/run_dl.py                       # fixed split, stocks
python src/dl/run_dl.py --split expanding
python src/dl/run_dl.py --dataset forex --models MLP LSTM22
```

On the DCE cluster (the full runs take too long for an interactive session):

```sh
mkdir -p logs
sbatch src/dl/dce_job.sh fixed
sbatch src/dl/dce_job.sh expanding
squeue -u $USER                               # empty = finished

sbatch src/dl/dce_job.sh rolling              # 2nd argument = dataset: forex, futures
python src/dl/dm_tests.py                     # after the runs: DM tests against both HAR baselines
```

## Output (`results/dl/`)

| File | Content |
|---|---|
| `<dataset>_<split>.csv` | one line per model and cutoff: `n_forecasts`, `QLIKE`, `MSE log RV`, `MSE RV`, DM statistic and p-value against pooled HAR, and the difference from pooled HAR in % |
| `<dataset>_<split>_by_year.csv` | the same losses for each test year |
| `dm_tests.csv` | from `dm_tests.py`: every model against pooled HAR and per-asset HAR, on QLIKE and on MSE log RV |
| `forecasts_<dataset>_<split>.csv.gz` | every forecast (not versioned, rebuilt by the script) |

In the `vs HAR (%)` columns, negative means better than pooled HAR. A negative DM statistic
with a p-value under 0.05 means significantly better on QLIKE.

## Results

All numbers below come from `results/dl/`. Change in MSE of `log(rv5)` against the
**pooled** logHAR-MJ (negative = better), and in brackets the DM p-value on QLIKE.

| Dataset | Split | Test from | Per-asset HAR | MLP | LSTM 22d | LSTM 66d |
|---|---|---|---|---|---|---|
| stocks | fixed | 2019 | +0.7% (0.34) | -0.2% (0.93) | +1.1% (0.27) | +0.1% (0.55) |
| stocks | fixed | 2021 | +0.5% (0.89) | +0.3% (0.08) | **-1.3% (0.002)** | **-1.5% (0.0001)** |
| stocks | expanding | 2019 | **-0.7% (0.0004)** | -0.4% (0.23) | +1.5% (0.37) | +1.4% (0.46) |
| stocks | expanding | 2021 | **-0.7% (0.00003)** | +0.2% (0.34) | **-1.1% (0.006)** | **-1.5% (0.001)** |
| stocks | rolling | 2019 | -0.2% (0.05) | -0.6% (0.28) | +1.3% (0.36) | +1.1% (0.41) |
| stocks | rolling | 2021 | -0.1% (0.05) | +0.1% (0.25) | -0.8% (0.65) | -1.4% (0.19) |
| forex | fixed | 2019 | -0.4% (0.04) | +0.1% (0.65) | +1.9% (0.008) | +1.6% (0.07) |
| forex | fixed | 2021 | +0.1% (0.11) | -0.5% (0.64) | +0.3% (0.23) | 0.0% (0.42) |
| forex | expanding | 2019 | -0.5% (0.03) | -0.5% (0.39) | +0.9% (0.06) | +0.7% (0.23) |
| forex | expanding | 2021 | -0.3% (0.03) | -0.4% (0.58) | +0.8% (0.10) | +0.6% (0.18) |
| forex | rolling | 2019 | +0.2% (0.51) | +1.4% (0.19) | +9.9% (0.0001) | +9.9% (0.0001) |
| forex | rolling | 2021 | +0.6% (0.26) | +0.5% (0.71) | +4.8% (0.002) | +4.7% (0.001) |
| futures | fixed | 2019 | -1.8% (0.01) | **-2.7% (0.000003)** | +0.9% (0.87) | +0.8% (0.97) |
| futures | fixed | 2021 | -1.5% (0.28) | **-1.9% (0.00004)** | -0.9% (0.05) | -1.1% (0.03) |
| futures | expanding | 2019 | -2.2% (0.005) | **-3.1% (0.00001)** | -0.6% (0.24) | -0.7% (0.22) |
| futures | expanding | 2021 | -1.7% (0.14) | **-3.0% (0.0002)** | **-2.1% (0.0005)** | **-2.3% (0.0004)** |
| futures | rolling | 2019 | -1.6% (0.08) | +0.7% (0.90) | +9.8% (0.002) | +9.7% (0.003) |
| futures | rolling | 2021 | -1.4% (0.30) | +1.1% (0.99) | +5.8% (0.05) | +5.8% (0.05) |

Bold: significantly better than pooled HAR (p < 0.01). The full set of tests, including every
model against **per-asset** HAR and on both losses, is in `dm_tests.csv`.

What the results say:

- **Against the strongest baseline (per-asset HAR), the networks rarely win.** The LSTM beats it
  on both losses only on stocks, test from 2021, fixed split (MSE -2.0%, p < 0.001). The MLP
  beats it on futures with fixed or expanding splits, by 0.7% to 1.5% in MSE, but with p between
  0.02 and 0.09: suggestive, not conclusive.
- **The LSTM needs a lot of data.** It does best on stocks (40 assets) and worst with the rolling
  window (1,000 days only). On forex and futures (5 assets each) with the rolling window it is
  5% to 10% worse than HAR.
- **The LSTM fails in the 2020 crash.** With the test from 2019 on stocks, the 66-day LSTM beats
  pooled HAR in 6 of 8 test years, but in 2020 its MSE is 9% to 17% higher. That one year turns
  the gain into a tie (`stocks_fixed_by_year.csv`, `stocks_expanding_by_year.csv`). The MLP does
  not have this problem.
- **On futures the MLP is the best model** in the fixed and expanding splits. Futures mix very
  different assets (energy, metals, grains), and both the MLP and per-asset HAR handle that better
  than one pooled regression.

Caveats: sigma2 comes from a single validation year, so QLIKE is noisier than MSE on
`log(rv5)`. With 18 settings and several models, a p-value around 0.02 to 0.05 can appear by
chance. Hyperparameters were fixed in advance and never tuned.

## Files

| File | Step |
|---|---|
| `src/dl/run_dl.py` | the pipeline: data, folds, baselines, MLP, LSTM, scores, CSVs |
| `src/dl/dce_job.sh` | batch job for the DCE cluster |
| `src/dl/dm_tests.py` | extra DM tests from the saved forecasts |
| `src/dl/stocks_har_mlp.py` | first script: HAR against several MLP sizes on the cleaned data |
| `notebooks/03_lstm.ipynb`, `04_lstm_grid.ipynb` | step-by-step LSTM, then the cutoff x window grid |
