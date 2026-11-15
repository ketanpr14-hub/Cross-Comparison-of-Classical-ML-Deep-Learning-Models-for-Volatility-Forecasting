# Tree models for one-day-ahead volatility

Do random forests and gradient boosting forecast tomorrow's realized variance better than
the HAR model? Short answer: on their own, no. Used as a correction of the HAR, yes on
futures, but most of that gain is a leverage effect that a linear model also captures.

## Protocol

Same protocol as the rest of the project. Data, cleaning and losses are imported from
`code/econometrics` (`data.py`, `evaluation.py`), not copied.

| | |
|---|---|
| Target | `log(rv5)` of the next session, per asset |
| Split | fixed: train before the cutoff, test after; cutoffs 2019-01-01 and 2021-01-01 |
| Validation | last year before the cutoff: LightGBM tree size and number of rounds only |
| Main loss | QLIKE on `rv5` (Patton, 2011); lower is better |
| Also | MSE on `log(rv5)`, MSE on `rv5`, Diebold-Mariano test against `logHAR-MJ`, 90 % Model Confidence Set |
| Test rows | identical to the other branches (stocks: 78,605 forecasts from 2019, 58,405 from 2021) |

A log forecast `f` becomes a variance forecast `h = exp(f + σ²/2)`, where `σ²` is the
variance of the model's log error estimated without the test set: training residuals for
the linear models, out-of-bag residuals for the forest, validation residuals for LightGBM.

## Models

| Name | Description |
|---|---|
| `logHAR-MJ` | benchmark: OLS of tomorrow's log RV on the daily, weekly and monthly log RV, the market log RV (day, week) and the jump share; one regression for all assets |
| `logHAR-MJ per asset` | the same, one regression per asset |
| `LHAR-MJ per asset` | the same plus yesterday's open-to-close return split into up and down moves (leverage, Corsi & Renò, 2012) |
| `RF` | random forest, 300 trees, leaves of at least 50 rows |
| `LGBM` | LightGBM on the log target |
| `HAR-MJ+LGBM` | LightGBM fitted on the residual of `logHAR-MJ`: the HAR does the linear part, the trees the rest |

Tree features: the six `logHAR-MJ` regressors, the positive and negative semivariances
(`x_pos`, `x_neg`), the log realized quarticity (`log_rq`) and yesterday's signed return
(`r`). All trees are fitted on all assets of a dataset together.

## Run it

```sh
cd code
python -m pip install -r ml/requirements.txt
python -m ml.checks                     # look-ahead checks, about 1 min
python -m ml.trees --dataset futures    # stocks | forex | futures | combined, 1 to 4 min
python -m ml.diagnose --dataset futures # after ml.trees: where the gain comes from
```

## Results

Test QLIKE (bold: best of the column; * : in the 90 % Model Confidence Set).

| Model | stocks 2019 | stocks 2021 | forex 2019 | forex 2021 | futures 2019 | futures 2021 |
|---|---|---|---|---|---|---|
| `logHAR-MJ` | 0.1818* | 0.1772 | 0.1531 | 0.1573* | 0.1731 | 0.1706 |
| `logHAR-MJ per asset` | **0.1817*** | 0.1777 | 0.1521* | 0.1571* | 0.1698 | 0.1683 |
| `LHAR-MJ per asset` | 0.1823* | 0.1783 | **0.1507*** | 0.1559* | 0.1631* | 0.1612* |
| `RF` | 0.1863* | 0.1791 | 0.1540* | 0.1564* | 0.1746 | 0.1640 |
| `LGBM` | 0.1900 | 0.1781 | 0.1564 | **0.1552*** | 0.1672 | **0.1583*** |
| `HAR-MJ+LGBM` | 0.1843* | **0.1756*** | 0.1525* | 0.1552* | **0.1618*** | 0.1585* |

Full tables (MSE, DM statistics, MCS p-values): `results/ml/<dataset>/metrics.csv`.

1. **Trees alone do not beat the HAR.** `RF` and `LGBM` are worse than `logHAR-MJ` on
   stocks and forex from 2019, and at best level with it elsewhere. Without a linear
   backbone, trees approximate a near-linear relation in steps.
2. **As a correction of the HAR, boosting helps on futures.** `HAR-MJ+LGBM` lowers QLIKE
   by 6.5 % and 7.1 % against `logHAR-MJ` (DM p < 0.001) and improves all five futures.
3. **On stocks the result is mixed.** Better from 2021 (-0.9 %, p = 0.002), worse from
   2019 (+1.4 %, p = 0.04); it beats the benchmark on 17 stocks out of 40.
4. **The combined panel** (50 assets) changes neither the ranking nor the conclusion
   (`results/ml/combined/`).

## Where the futures gain comes from

The five futures are unrelated markets: corn (C), crude oil (CL), S&P 500 e-mini (ES),
gold (GC) and natural gas (NG). `ml.diagnose` separates three sources.

**Gain against `logHAR-MJ`, per asset** (QLIKE change, both cutoffs; `per_asset.csv`):

| | C | CL | ES | GC | NG |
|---|---|---|---|---|---|
| `logHAR-MJ per asset` | -5.2 % | -2.8 % | -1.7 % | +2.1 % | +2.0 % |
| `LHAR-MJ per asset` | -5.1 % | -8.0 % | -8.5 % | -1.3 % | -4.0 % |
| `HAR-MJ+LGBM` | -6.0 % | -11.6 % | -9.7 % | -2.1 % | -2.5 % |

- **Heterogeneity.** One regression for five different markets fits corn badly. One
  regression per asset recovers almost all of the corn gain.
- **Leverage.** For ES and CL the gain comes from the sign of yesterday's return. Error of
  `logHAR-MJ` after a one-standard-deviation move (`leverage.csv`, cutoff 2021):

  | | C | CL | ES | GC | NG |
  |---|---|---|---|---|---|
  | after an up move | -0.03 | +0.04 | -0.05 | +0.03 | +0.05 |
  | after a down move | +0.04 | **+0.17** | **+0.25** | +0.06 | +0.01 |

  A down day raises tomorrow's variance far more than an up day of the same size, mostly
  for the equity index and oil. The HAR only sees the size of past moves, not their sign.
  This is why `r` is the first feature of `HAR-MJ+LGBM` (17 % of the total gain,
  `importance.csv`), and why adding `r` alone yields the largest part of the gain
  (`ablation.csv`: 0.1699 → 0.1657 from 2019, out of 0.1618 with all features).
- **Non-linearity.** A linear HAR per asset with the two leverage terms (`LHAR-MJ per
  asset`) captures most of the gain: 0.1631 and 0.1612, against 0.1618 and 0.1585 for
  `HAR-MJ+LGBM`. What is left for the trees is about 1 % to 2 % of QLIKE.

On stocks, the leverage terms do not help even linearly (0.1823 against 0.1818), although
`r` is also the first feature of the trees there. A likely reason, not tested here: the
market-wide level (`m_day`) already carries most of the information of an individual
stock's down move.

## Accuracy in plain terms

Averages over both cutoffs (`plain.csv`). RW = tomorrow equals today.

| | Model | R² on log RV | Typical error on volatility | Forecasts within ±20 % | Right direction |
|---|---|---|---|---|---|
| stocks | RW | 0.50 | 18.8 % | 52.7 % | – |
| | `logHAR-MJ` | 0.64 | 15.9 % | 60.1 % | 67.9 % |
| | `HAR-MJ+LGBM` | 0.64 | 15.8 % | 60.6 % | 68.1 % |
| futures | RW | 0.79 | 16.7 % | 57.7 % | – |
| | `logHAR-MJ` | 0.84 | 14.5 % | 63.5 % | 66.4 % |
| | `HAR-MJ+LGBM` | 0.85 | 14.0 % | 65.4 % | 67.1 % |

- Typical error: median of |forecast volatility / realized volatility − 1|.
- Right direction: share of days where the model correctly says whether volatility goes up
  or down compared with today.
- The futures R² is high because the five assets have very different volatility levels,
  and those level differences are easy to predict.
- Most of the improvement comes from the HAR structure itself. The remaining error,
  about 15 % on daily volatility, is largely the measurement noise of a single day.

## Checks

`python -m ml.checks` asserts, on the four datasets and both cutoffs:
- every training label is known at the first test origin (the close of that day);
- no test target is ever used as a training label, and the same holds between the fit
  part and the validation year;
- the target is `y(t+1)` and the weekly and monthly lags end at `t`, recomputed from the
  raw file.

Two limits that hold for every branch:
- the 40 stocks are today's index members, so there is some survivorship bias;
- futures roll days are masked in the test set too, so they are not scored.

## Files

| File | Content |
|---|---|
| `trees.py` | models, validation grid, scoring; writes `metrics.csv`, `tuning.csv`, `importance.csv`, `forecasts.csv.gz` (not versioned) |
| `diagnose.py` | `per_asset.csv`, `leverage.csv`, `ablation.csv`, `plain.csv` |
| `checks.py` | look-ahead checks |

## References

- Corsi (2009), *A simple approximate long-memory model of realized volatility*, J. Financial Econometrics.
- Corsi & Renò (2012), *Discrete-time volatility forecasting with persistent leverage effect*, J. Business & Economic Statistics.
- Patton (2011), *Volatility forecast comparison using imperfect volatility proxies*, J. Econometrics.
- Hansen, Lunde & Nason (2011), *The Model Confidence Set*, Econometrica.
- Christensen, Siggaard & Veliyev (2023), *A machine learning approach to volatility forecasting*, J. Financial Econometrics.
- Branco, Rubesam & Zevallos (2024), *Forecasting realized volatility: Does anything beat linear models?*, J. Empirical Finance.
