# Econometric volatility forecasts: stocks

Run `20261114-165326_fixed_tuned` (2026-11-14T16:53:55, 201.1 s, commit f59e035).

## Settings

- **Dataset**: stocks, 40 assets, 2015-01-02 to 2026-10-30 (119,000 asset-days, 6 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: fixed: one estimation on all data before the cutoff; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: selected by QLIKE on 2 validation year(s) before each cutoff, then the model is re-estimated on the whole training period (values in the tables below)
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

78,605 forecasts on 1,968 days (2019-01-02 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1811 |           -2.8014 |   -3.0338 |       0.0024 |              16 |              1 |        1.0000 | yes      |       0.2851 | 2.262e-07 |
|      2 | logSHAR   |  0.1859 |           -0.2587 |   -1.6747 |       0.0940 |               7 |              0 |        0.0030 |          |       0.2903 | 2.310e-07 |
|      3 | logHAR    |  0.1863 |            0.0000 |           |              |                 |                |        0.0030 |          |       0.2910 | 2.325e-07 |
|      4 | HAR       |  0.1916 |            2.8441 |    2.5794 |       0.0099 |               0 |             13 |        0.0000 |          |       0.3234 | 2.502e-07 |
|      5 | MEM       |  0.1945 |            4.3634 |    2.9466 |       0.0032 |               0 |             20 |        0.0000 |          |       0.3309 | 2.655e-07 |
|      6 | HARQ      |  0.1958 |            5.0599 |    2.7547 |       0.0059 |               0 |             10 |        0.0000 |          |       0.3277 | 2.289e-07 |
|      7 | AMEM      |  0.1963 |            5.3592 |    3.4340 |       0.0006 |               0 |             20 |        0.0000 |          |       0.3330 | 2.693e-07 |
|      8 | EWMA-RV   |  0.2052 |           10.1055 |    6.1460 |       0.0000 |               0 |             37 |        0.0000 |          |       0.3339 | 2.467e-07 |
|      9 | RW        |  0.2486 |           33.4099 |   14.8176 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3860 | 3.309e-07 |
|     10 | GARCH     |  0.2724 |           46.2037 |    4.8747 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4295 | 3.039e-07 |
|     11 | GJR-GARCH |  0.2812 |           50.8787 |    4.9456 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4326 | 3.087e-07 |
|     12 | EGARCH    |  0.2852 |           53.0621 |    3.9822 |       0.0001 |               0 |             37 |        0.0005 |          |       0.4288 | 6.017e-07 |
|     13 | EWMA      |  0.2928 |           57.1399 |    3.3216 |       0.0009 |               0 |             40 |        0.0030 |          |       0.4758 | 3.187e-07 |
|     14 | ARCH      |  0.3052 |           63.7983 |    6.9454 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4712 | 3.063e-07 |

Hyperparameters used:

| model     | hyperparameters                   |   candidates |   validation QLIKE |   with default values |
|:----------|:----------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                              |            1 |                    |                       |
| EWMA      | lambda = 0.97                     |            6 |             0.2845 |                0.2862 |
| EWMA-RV   | none                              |            1 |                    |                       |
| ARCH      | p = 22, dist = t                  |           10 |             0.2772 |                0.3066 |
| GARCH     | p = 1, q = 2, dist = t            |            8 |             0.2712 |                0.2776 |
| GJR-GARCH | p = 1, q = 2, dist = t            |            4 |             0.2666 |                0.2761 |
| EGARCH    | p = 1, q = 2, dist = t            |            4 |           147.8916 |            11086.0389 |
| HAR       | pooling = pooled, estimator = wls |            4 |             0.1929 |                0.2227 |
| HARQ      | pooling = pooled, estimator = wls |            4 |             0.1952 |                0.2044 |
| logHAR    | pooling = pooled                  |            2 |             0.1897 |                0.1897 |
| logHAR-MJ | pooling = per_asset               |            2 |             0.1809 |                0.1828 |
| logSHAR   | pooling = pooled                  |            2 |             0.1898 |                0.1898 |
| MEM       | p = 2, q = 2                      |            4 |             0.1962 |                0.1988 |
| AMEM      | p = 2, q = 2                      |            4 |             0.1972 |                0.1995 |

## Test from 2021-01-01

58,405 forecasts on 1,463 days (2021-01-04 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1766 |           -2.5429 |   -3.0902 |       0.0020 |              19 |              2 |        1.0000 | yes      |       0.2749 | 1.659e-07 |
|      2 | logSHAR   |  0.1805 |           -0.4049 |   -3.3911 |       0.0007 |              11 |              1 |        0.0085 |          |       0.2805 | 1.663e-07 |
|      3 | logHAR    |  0.1813 |            0.0000 |           |              |                 |                |        0.0045 |          |       0.2821 | 1.671e-07 |
|      4 | HAR       |  0.1878 |            3.6185 |    4.8702 |       0.0000 |               0 |             25 |        0.0000 |          |       0.3156 | 1.994e-07 |
|      5 | MEM       |  0.1887 |            4.0895 |    5.1803 |       0.0000 |               0 |             31 |        0.0000 |          |       0.3254 | 2.332e-07 |
|      6 | AMEM      |  0.1892 |            4.3686 |    5.1209 |       0.0000 |               0 |             30 |        0.0000 |          |       0.3254 | 2.393e-07 |
|      7 | HARQ      |  0.1979 |            9.2056 |    3.4996 |       0.0005 |               0 |             31 |        0.0005 |          |       0.3271 | 1.753e-07 |
|      8 | EWMA-RV   |  0.2021 |           11.4765 |   11.3358 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3249 | 2.044e-07 |
|      9 | EGARCH    |  0.2332 |           28.6288 |   11.3587 |       0.0000 |               0 |             37 |        0.0000 |          |       0.3899 | 1.870e-07 |
|     10 | GARCH     |  0.2431 |           34.1266 |   10.8527 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4035 | 1.913e-07 |
|     11 | GJR-GARCH |  0.2443 |           34.7943 |   12.6832 |       0.0000 |               0 |             38 |        0.0000 |          |       0.4013 | 1.924e-07 |
|     12 | RW        |  0.2477 |           36.6315 |   17.3965 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3817 | 2.862e-07 |
|     13 | ARCH      |  0.2661 |           46.8072 |   15.8955 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4376 | 1.963e-07 |
|     14 | EWMA      |  0.2726 |           50.4103 |   13.4917 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4285 | 1.930e-07 |

Hyperparameters used:

| model     | hyperparameters                      |   candidates |   validation QLIKE |   with default values |
|:----------|:-------------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                                 |            1 |                    |                       |
| EWMA      | lambda = 0.9                         |            6 |             0.3364 |                0.3562 |
| EWMA-RV   | none                                 |            1 |                    |                       |
| ARCH      | p = 22, dist = t                     |           10 |             0.3694 |                0.4483 |
| GARCH     | p = 1, q = 1, dist = t               |            8 |             0.3387 |                0.3524 |
| GJR-GARCH | p = 1, q = 2, dist = t               |            4 |             0.3570 |                0.3762 |
| EGARCH    | p = 1, q = 2, dist = t               |            4 |             0.3894 |                0.4215 |
| HAR       | pooling = per_asset, estimator = wls |            4 |             0.1979 |                0.2266 |
| HARQ      | pooling = pooled, estimator = wls    |            4 |             0.1980 |                0.2095 |
| logHAR    | pooling = pooled                     |            2 |             0.2023 |                0.2023 |
| logHAR-MJ | pooling = per_asset                  |            2 |             0.1948 |                0.1967 |
| logSHAR   | pooling = pooled                     |            2 |             0.2022 |                0.2022 |
| MEM       | p = 1, q = 2                         |            4 |             0.1960 |                0.1972 |
| AMEM      | p = 1, q = 2                         |            4 |             0.1980 |                0.1994 |

## How to read the table

- **QLIKE** is the loss to minimize (0 = perfect forecast). Models are ranked on it.
- **DM stat / p-value**: Diebold-Mariano test against logHAR on the daily loss averaged across assets. A negative statistic means the model beats the benchmark; the difference is significant at 5 % when the p-value is below 0.05.
- **assets better / worse**: number of assets (out of 40) where the model is significantly better / worse than the benchmark at 5 % when the test is run asset by asset.
- **in MCS**: the model belongs to the Model Confidence Set, the smallest group of models that contains the best one with the stated confidence. Models outside it are significantly worse than the best.
- **MSE log RV** is the mean squared error on log(rv5), the criterion of the DL and foundation-model branches, on the same asset-days when the dataset is stocks.
- **MSE RV** is the mean squared error on rv5 itself. A handful of crisis days carry most of it, so it is much noisier than QLIKE.

## Figures

![QLIKE ranking](figures/qlike_ranking.png)

![Diebold-Mariano matrix](figures/dm_matrix.png)

![Forecast path](figures/forecast_path.png)
