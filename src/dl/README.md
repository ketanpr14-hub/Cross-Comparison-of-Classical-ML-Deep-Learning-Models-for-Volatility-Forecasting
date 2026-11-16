# Deep learning models for one-day-ahead volatility (MLP, LSTM)

This part of the project tests whether a small neural network forecasts tomorrow's realized
variance better than the HAR regression of the econometrics branch. It uses Jean's
`build_panel` (in `code/econometrics/`), so the cleaning, the features and the test rows are
the same as his: 78,605 forecasts from 2019 and 58,405 from 2021 on the 40 stocks.

## Protocol

| | |
|---|---|
| Target | `log(rv5)` of the next session, asset by asset |
| Cutoffs | test from 2019-01-01 and from 2021-01-01 |
| Splits | `fixed`: train once before the cutoff. `expanding`: refit every 252 days on all past data (Jean's folds). `rolling` is available but not run |
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

Python 3.12, VOLARE folders under `data/` at the project root (not versioned).

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
```

## Output (`results/dl/`)

| File | Content |
|---|---|
| `stocks_<split>.csv` | one line per model and cutoff: `n_forecasts`, `QLIKE`, `MSE log RV`, `MSE RV`, DM statistic and p-value against pooled HAR, and the difference from pooled HAR in % |
| `stocks_<split>_by_year.csv` | the same losses for each test year |
| `forecasts_stocks_<split>.csv.gz` | every forecast (not versioned, rebuilt by the script) |

In the `vs HAR (%)` columns, negative means better than pooled HAR. A negative DM statistic
with a p-value under 0.05 means significantly better on QLIKE.

## Results on stocks

MSE on `log(rv5)` and QLIKE; in brackets the DM p-value against pooled HAR.

| Cutoff | Split | pooled HAR | per-asset HAR | MLP | LSTM 22d | LSTM 66d |
|---|---|---|---|---|---|---|
| 2019 | fixed | 0.2831 / 0.1820 | 0.2851 / 0.1811 (0.34) | 0.2826 / 0.1819 (0.93) | 0.2864 / 0.1870 (0.27) | 0.2834 / 0.1843 (0.55) |
| 2019 | expanding | 0.2836 / 0.1825 | 0.2816 / 0.1801 (0.0004) | 0.2825 / 0.1820 (0.23) | 0.2879 / 0.1874 (0.37) | 0.2877 / 0.1868 (0.46) |
| 2021 | fixed | 0.2735 / 0.1766 | 0.2749 / 0.1766 (0.89) | 0.2743 / 0.1771 (0.08) | 0.2698 / 0.1748 (0.002) | 0.2693 / 0.1745 (0.00008) |
| 2021 | expanding | 0.2729 / 0.1773 | 0.2711 / 0.1746 (0.00003) | 0.2734 / 0.1776 (0.34) | 0.2699 / 0.1761 (0.006) | 0.2688 / 0.1751 (0.001) |

- The MLP is never significantly different from pooled HAR.
- The LSTM is significantly better than pooled HAR when the test starts in 2021 (about 1.1% to
  1.5% lower MSE), and tied with it when the test starts in 2019.
- With yearly refits, per-asset HAR is the strongest baseline. From 2021 the 66-day LSTM has a
  lower MSE than it (0.2688 against 0.2711) but a slightly higher QLIKE (0.1751 against
  0.1746). That pair has not been tested with a DM test.
- Refitting every year barely changes the networks.

Caveats: sigma2 comes from a single validation year (2020 for the 2021 cutoff), so QLIKE is
noisier than MSE on `log(rv5)`. Forex and futures have not been run yet.

## Files

| File | Step |
|---|---|
| `src/dl/run_dl.py` | the pipeline: data, folds, baselines, MLP, LSTM, scores, CSVs |
| `src/dl/dce_job.sh` | batch job for the DCE cluster |
| `src/dl/stocks_har_mlp.py` | first script: HAR against several MLP sizes on the cleaned data |
| `notebooks/03_lstm.ipynb`, `04_lstm_grid.ipynb` | step-by-step LSTM, then the cutoff x window grid |
