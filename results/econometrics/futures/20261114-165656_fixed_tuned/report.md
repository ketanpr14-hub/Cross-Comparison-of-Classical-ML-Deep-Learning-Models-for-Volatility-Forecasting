# Econometric volatility forecasts: futures

Run `20261114-165656_fixed_tuned` (2026-11-14T16:57:00, 78.1 s, commit f59e035).

## Settings

- **Dataset**: futures, 5 assets, 2009-09-28 to 2026-10-30 (22,028 asset-days, 36 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: fixed: one estimation on all data before the cutoff; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: selected by QLIKE on 2 validation year(s) before each cutoff, then the model is re-estimated on the whole training period (values in the tables below)
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

9,778 forecasts on 2,026 days (2019-01-02 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | AMEM      |  0.1674 |           -1.4352 |   -1.3641 |       0.1725 |               1 |              0 |        1.0000 | yes      |       0.3123 | 6.695e-07 |
|      2 | logSHAR   |  0.1683 |           -0.9155 |   -2.8588 |       0.0043 |               2 |              0 |        0.5240 | yes      |       0.2843 | 6.363e-07 |
|      3 | MEM       |  0.1692 |           -0.3608 |   -0.5388 |       0.5900 |               0 |              0 |        0.0930 |          |       0.3168 | 6.643e-07 |
|      4 | logHAR-MJ |  0.1695 |           -0.2012 |   -1.2003 |       0.2300 |               1 |              1 |        0.2355 | yes      |       0.2858 | 6.315e-07 |
|      5 | logHAR    |  0.1698 |            0.0000 |           |              |                 |                |        0.0435 |          |       0.2880 | 6.371e-07 |
|      6 | HAR       |  0.1728 |            1.7652 |    1.0959 |       0.2731 |               0 |              1 |        0.0255 |          |       0.3285 | 6.755e-07 |
|      7 | EWMA-RV   |  0.1759 |            3.5730 |    2.8541 |       0.0043 |               0 |              2 |        0.0000 |          |       0.3191 | 6.683e-07 |
|      8 | GARCH     |  0.2115 |           24.5438 |    7.4616 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4191 | 9.474e-07 |
|      9 | GJR-GARCH |  0.2196 |           29.2956 |    5.8460 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4119 | 1.105e-06 |
|     10 | EGARCH    |  0.2211 |           30.1993 |    4.1282 |       0.0000 |               0 |              4 |        0.0005 |          |       0.3750 | 7.452e-07 |
|     11 | ARCH      |  0.2301 |           35.4554 |   11.7279 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4456 | 1.629e-06 |
|     12 | EWMA      |  0.2315 |           36.2946 |    7.9418 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4447 | 1.010e-06 |
|     13 | HARQ      |  0.2404 |           41.5739 |    1.3851 |       0.1660 |               0 |              2 |        0.2355 | yes      |       0.3700 | 6.340e-07 |
|     14 | RW        |  0.2582 |           52.0127 |    9.9865 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3888 | 9.761e-07 |

Hyperparameters used:

| model     | hyperparameters                   |   candidates |   validation QLIKE |   with default values |
|:----------|:----------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                              |            1 |                    |                       |
| EWMA      | lambda = 0.94                     |            6 |             0.2013 |                0.2013 |
| EWMA-RV   | none                              |            1 |                    |                       |
| ARCH      | p = 22, dist = t                  |           10 |             0.2314 |                0.2874 |
| GARCH     | p = 1, q = 2, dist = t            |            8 |             0.2065 |                0.2138 |
| GJR-GARCH | p = 1, q = 2, dist = t            |            4 |             0.1977 |                0.2057 |
| EGARCH    | p = 1, q = 2, dist = t            |            4 |             0.1914 |                0.1988 |
| HAR       | pooling = pooled, estimator = wls |            4 |             0.1571 |                0.2018 |
| HARQ      | pooling = pooled, estimator = ols |            4 |             0.1920 |                0.1920 |
| logHAR    | pooling = per_asset               |            2 |             0.1531 |                0.1547 |
| logHAR-MJ | pooling = per_asset               |            2 |             0.1525 |                0.1562 |
| logSHAR   | pooling = per_asset               |            2 |             0.1508 |                0.1532 |
| MEM       | p = 2, q = 2                      |            4 |             0.1633 |                0.1673 |
| AMEM      | p = 2, q = 2                      |            4 |             0.1627 |                0.1661 |

## Test from 2021-01-01

7,289 forecasts on 1,509 days (2021-01-04 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logSHAR   |  0.1661 |           -1.0913 |   -2.6973 |       0.0070 |               2 |              0 |        1.0000 | yes      |       0.2797 | 6.292e-07 |
|      2 | AMEM      |  0.1672 |           -0.4397 |   -0.7385 |       0.4602 |               1 |              1 |        0.7920 | yes      |       0.3114 | 7.057e-07 |
|      3 | logHAR-MJ |  0.1680 |           -0.0114 |   -0.5916 |       0.5541 |               1 |              1 |        0.1985 | yes      |       0.2817 | 6.246e-07 |
|      4 | logHAR    |  0.1680 |            0.0000 |           |              |                 |                |        0.0640 |          |       0.2837 | 6.301e-07 |
|      5 | HAR       |  0.1686 |            0.3750 |   -0.0031 |       0.9975 |               0 |              0 |        0.5495 | yes      |       0.3088 | 6.903e-07 |
|      6 | MEM       |  0.1689 |            0.5674 |    0.2372 |       0.8125 |               0 |              1 |        0.3105 | yes      |       0.3148 | 6.938e-07 |
|      7 | EWMA-RV   |  0.1751 |            4.2488 |    3.3743 |       0.0007 |               0 |              2 |        0.0000 |          |       0.3163 | 6.860e-07 |
|      8 | GARCH     |  0.1928 |           14.7840 |    5.7503 |       0.0000 |               0 |              4 |        0.0000 |          |       0.3788 | 6.666e-07 |
|      9 | GJR-GARCH |  0.1976 |           17.6108 |    4.2772 |       0.0000 |               0 |              3 |        0.0010 |          |       0.3663 | 7.079e-07 |
|     10 | EGARCH    |  0.2056 |           22.3941 |    3.1785 |       0.0015 |               0 |              3 |        0.0205 |          |       0.3651 | 6.710e-07 |
|     11 | EWMA      |  0.2164 |           28.8096 |    8.2731 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3995 | 6.958e-07 |
|     12 | ARCH      |  0.2225 |           32.4914 |    9.7684 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4270 | 7.244e-07 |
|     13 | HARQ      |  0.2556 |           52.1921 |    1.2679 |       0.2048 |               0 |              1 |        0.4415 | yes      |       0.3356 | 6.301e-07 |
|     14 | RW        |  0.2601 |           54.8745 |    8.5346 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3879 | 1.029e-06 |

Hyperparameters used:

| model     | hyperparameters                      |   candidates |   validation QLIKE |   with default values |
|:----------|:-------------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                                 |            1 |                    |                       |
| EWMA      | lambda = 0.9                         |            6 |             0.2769 |                0.3060 |
| EWMA-RV   | none                                 |            1 |                    |                       |
| ARCH      | p = 10, dist = normal                |           10 |             0.2605 |                0.3216 |
| GARCH     | p = 1, q = 2, dist = normal          |            8 |             0.2641 |                0.2648 |
| GJR-GARCH | p = 1, q = 2, dist = normal          |            4 |             0.2715 |                0.2726 |
| EGARCH    | p = 1, q = 2, dist = t               |            4 |             0.2703 |                0.2723 |
| HAR       | pooling = per_asset, estimator = wls |            4 |             0.1678 |                0.1944 |
| HARQ      | pooling = pooled, estimator = wls    |            4 |             0.1738 |                0.1825 |
| logHAR    | pooling = per_asset                  |            2 |             0.1704 |                0.1769 |
| logHAR-MJ | pooling = per_asset                  |            2 |             0.1692 |                0.1777 |
| logSHAR   | pooling = per_asset                  |            2 |             0.1693 |                0.1762 |
| MEM       | p = 2, q = 2                         |            4 |             0.1870 |                0.1902 |
| AMEM      | p = 2, q = 2                         |            4 |             0.1835 |                0.1858 |

## How to read the table

- **QLIKE** is the loss to minimize (0 = perfect forecast). Models are ranked on it.
- **DM stat / p-value**: Diebold-Mariano test against logHAR on the daily loss averaged across assets. A negative statistic means the model beats the benchmark; the difference is significant at 5 % when the p-value is below 0.05.
- **assets better / worse**: number of assets (out of 5) where the model is significantly better / worse than the benchmark at 5 % when the test is run asset by asset.
- **in MCS**: the model belongs to the Model Confidence Set, the smallest group of models that contains the best one with the stated confidence. Models outside it are significantly worse than the best.
- **MSE log RV** is the mean squared error on log(rv5), the criterion of the DL and foundation-model branches, on the same asset-days when the dataset is stocks.
- **MSE RV** is the mean squared error on rv5 itself. A handful of crisis days carry most of it, so it is much noisier than QLIKE.

## Figures

![QLIKE ranking](figures/qlike_ranking.png)

![Diebold-Mariano matrix](figures/dm_matrix.png)

![Forecast path](figures/forecast_path.png)
