# Econometric volatility forecasts: stocks

Run `20261114-180457_expanding_tuned` (2026-11-14T18:05:25, 4095.3 s, commit f59e035).

## Settings

- **Dataset**: stocks, 40 assets, 2015-01-02 to 2026-10-30 (119,000 asset-days, 6 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: expanding: re-estimated every 252 days on all past data; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: selected by QLIKE on 2 validation year(s) before each cutoff, then the model is re-estimated on the whole training period (values in the tables below)
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

78,605 forecasts on 1,968 days (2019-01-02 to 2026-10-29), 8 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1801 |           -3.5917 |   -4.1523 |       0.0000 |              19 |              0 |        1.0000 | yes      |       0.2816 | 2.258e-07 |
|      2 | logSHAR   |  0.1863 |           -0.2923 |   -2.1796 |       0.0293 |               8 |              1 |        0.0000 |          |       0.2908 | 2.326e-07 |
|      3 | logHAR    |  0.1868 |            0.0000 |           |              |                 |                |        0.0000 |          |       0.2916 | 2.341e-07 |
|      4 | MEM       |  0.1897 |            1.5273 |    1.0496 |       0.2939 |               0 |              5 |        0.0000 |          |       0.3295 | 2.713e-07 |
|      5 | AMEM      |  0.1903 |            1.8973 |    1.2595 |       0.2079 |               0 |              8 |        0.0000 |          |       0.3304 | 2.743e-07 |
|      6 | HAR       |  0.1913 |            2.4129 |    2.0932 |       0.0363 |               0 |              9 |        0.0000 |          |       0.3270 | 2.531e-07 |
|      7 | HARQ      |  0.1969 |            5.4038 |    2.4648 |       0.0137 |               0 |              8 |        0.0000 |          |       0.3313 | 2.323e-07 |
|      8 | EWMA-RV   |  0.2032 |            8.8031 |    4.9873 |       0.0000 |               0 |             33 |        0.0000 |          |       0.3309 | 2.466e-07 |
|      9 | RW        |  0.2486 |           33.0844 |   14.1916 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3860 | 3.309e-07 |
|     10 | GARCH     |  0.2626 |           40.5808 |    4.3403 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4221 | 3.007e-07 |
|     11 | GJR-GARCH |  0.2684 |           43.7042 |    4.1883 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4238 | 3.005e-07 |
|     12 | EGARCH    |  0.2695 |           44.2743 |    3.3101 |       0.0009 |               0 |             35 |        0.0030 |          |       0.4170 | 3.057e-07 |
|     13 | ARCH      |  0.2851 |           52.5986 |    5.2477 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4498 | 3.018e-07 |
|     14 | EWMA      |  0.2928 |           56.7566 |    3.3296 |       0.0009 |               0 |             40 |        0.0030 |          |       0.4758 | 3.187e-07 |

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

58,405 forecasts on 1,463 days (2021-01-04 to 2026-10-29), 6 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1746 |           -3.5795 |   -4.1750 |       0.0000 |              19 |              1 |        1.0000 | yes      |       0.2711 | 1.652e-07 |
|      2 | logSHAR   |  0.1803 |           -0.3879 |   -3.0296 |       0.0024 |              11 |              1 |        0.0020 |          |       0.2796 | 1.664e-07 |
|      3 | logHAR    |  0.1811 |            0.0000 |           |              |                 |                |        0.0020 |          |       0.2811 | 1.671e-07 |
|      4 | HAR       |  0.1860 |            2.7611 |    3.5689 |       0.0004 |               0 |             24 |        0.0000 |          |       0.3143 | 1.978e-07 |
|      5 | MEM       |  0.1870 |            3.2927 |    4.1212 |       0.0000 |               0 |             25 |        0.0000 |          |       0.3242 | 2.391e-07 |
|      6 | AMEM      |  0.1872 |            3.3956 |    3.9391 |       0.0001 |               0 |             25 |        0.0000 |          |       0.3238 | 2.427e-07 |
|      7 | HARQ      |  0.1961 |            8.3148 |    3.1564 |       0.0016 |               0 |             24 |        0.0005 |          |       0.3248 | 1.768e-07 |
|      8 | EWMA-RV   |  0.2012 |           11.1122 |   11.2829 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3244 | 2.042e-07 |
|      9 | EGARCH    |  0.2281 |           25.9770 |    9.1416 |       0.0000 |               0 |             37 |        0.0000 |          |       0.3836 | 1.794e-07 |
|     10 | GARCH     |  0.2375 |           31.1900 |    9.2919 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4021 | 1.884e-07 |
|     11 | GJR-GARCH |  0.2378 |           31.3320 |   10.5332 |       0.0000 |               0 |             38 |        0.0000 |          |       0.3981 | 1.880e-07 |
|     12 | RW        |  0.2477 |           36.7888 |   16.6707 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3817 | 2.862e-07 |
|     13 | ARCH      |  0.2558 |           41.2729 |   13.7144 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4274 | 1.938e-07 |
|     14 | EWMA      |  0.2726 |           50.5835 |   13.8388 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4285 | 1.930e-07 |

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
