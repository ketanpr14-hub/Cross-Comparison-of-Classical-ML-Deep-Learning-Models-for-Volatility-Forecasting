# Econometric volatility forecasts: stocks

Run `20261113-210212_expanding` (2026-11-13T21:02:33, 64.0 s, commit 266fffe).

## Settings

- **Dataset**: stocks, 40 assets, 2015-01-02 to 2026-10-30 (119,000 asset-days, 6 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: expanding: re-estimated every 252 days on all past data; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: EWMA lambda 0.94, ARCH order 5, GARCH errors normal, HAR estimation pooled with lags 1/5/22
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

78,605 forecasts on 1,968 days (2019-01-02 to 2026-10-29), 8 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1825 |           -2.2966 |   -2.6780 |       0.0074 |              19 |              0 |        1.0000 | yes      |       0.2836 | 2.246e-07 |
|      2 | logSHAR   |  0.1863 |           -0.2923 |   -2.1796 |       0.0293 |               8 |              1 |        0.0110 |          |       0.2908 | 2.326e-07 |
|      3 | logHAR    |  0.1868 |            0.0000 |           |              |                 |                |        0.0110 |          |       0.2916 | 2.341e-07 |
|      4 | MEM       |  0.1903 |            1.8810 |    1.2604 |       0.2075 |               0 |              8 |        0.0000 |          |       0.3308 | 2.669e-07 |
|      5 | AMEM      |  0.1910 |            2.2335 |    1.4485 |       0.1475 |               0 |             10 |        0.0000 |          |       0.3315 | 2.699e-07 |
|      6 | HARQ      |  0.1995 |            6.8086 |    3.3927 |       0.0007 |               0 |             19 |        0.0000 |          |       0.3478 | 2.396e-07 |
|      7 | HAR       |  0.2017 |            7.9607 |    8.8567 |       0.0000 |               0 |             36 |        0.0000 |          |       0.3671 | 2.541e-07 |
|      8 | EWMA-RV   |  0.2032 |            8.8031 |    4.9873 |       0.0000 |               0 |             33 |        0.0000 |          |       0.3309 | 2.466e-07 |
|      9 | RW        |  0.2486 |           33.0844 |   14.1916 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3860 | 3.309e-07 |
|     10 | GARCH     |  0.2681 |           43.5243 |    4.3329 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4223 | 3.012e-07 |
|     11 | GJR-GARCH |  0.2769 |           48.2138 |    4.3404 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4285 | 3.031e-07 |
|     12 | EWMA      |  0.2780 |           48.7975 |    5.3614 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4442 | 3.020e-07 |
|     13 | EGARCH    |  0.2813 |           50.6138 |    3.2903 |       0.0010 |               0 |             35 |        0.0050 |          |       0.4229 | 3.217e-07 |
|     14 | ARCH      |  0.3177 |           70.0734 |    5.1226 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4780 | 3.139e-07 |

## Test from 2021-01-01

58,405 forecasts on 1,463 days (2021-01-04 to 2026-10-29), 6 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1772 |           -2.1366 |   -2.5449 |       0.0109 |              17 |              0 |        1.0000 | yes      |       0.2729 | 1.652e-07 |
|      2 | logSHAR   |  0.1803 |           -0.3879 |   -3.0296 |       0.0024 |              11 |              1 |        0.0265 |          |       0.2796 | 1.664e-07 |
|      3 | logHAR    |  0.1811 |            0.0000 |           |              |                 |                |        0.0115 |          |       0.2811 | 1.671e-07 |
|      4 | MEM       |  0.1875 |            3.5783 |    4.1390 |       0.0000 |               0 |             24 |        0.0000 |          |       0.3255 | 2.354e-07 |
|      5 | AMEM      |  0.1877 |            3.6461 |    3.9474 |       0.0001 |               0 |             26 |        0.0000 |          |       0.3251 | 2.393e-07 |
|      6 | HAR       |  0.1928 |            6.5095 |    8.6366 |       0.0000 |               0 |             26 |        0.0000 |          |       0.3498 | 1.841e-07 |
|      7 | HARQ      |  0.1957 |            8.1064 |    3.1130 |       0.0019 |               0 |             19 |        0.0030 |          |       0.3354 | 1.754e-07 |
|      8 | EWMA-RV   |  0.2012 |           11.1122 |   11.2829 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3244 | 2.042e-07 |
|      9 | EGARCH    |  0.2329 |           28.6579 |    8.4720 |       0.0000 |               0 |             37 |        0.0000 |          |       0.3858 | 1.966e-07 |
|     10 | GARCH     |  0.2389 |           31.9681 |    8.9869 |       0.0000 |               0 |             39 |        0.0000 |          |       0.3979 | 1.867e-07 |
|     11 | GJR-GARCH |  0.2425 |           33.9405 |    9.8325 |       0.0000 |               0 |             39 |        0.0000 |          |       0.4002 | 1.892e-07 |
|     12 | RW        |  0.2477 |           36.7888 |   16.6707 |       0.0000 |               0 |             40 |        0.0000 |          |       0.3817 | 2.862e-07 |
|     13 | EWMA      |  0.2508 |           38.5161 |    9.4312 |       0.0000 |               0 |             40 |        0.0000 |          |       0.4123 | 1.879e-07 |
|     14 | ARCH      |  0.2726 |           50.5397 |   14.9551 |       0.0000 |               0 |             38 |        0.0000 |          |       0.4483 | 2.019e-07 |

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
