# Econometric volatility forecasts: futures

Run `20261113-194058_fixed` (2026-11-13T19:40:59, 10.9 s, commit 80dc556).

## Settings

- **Dataset**: futures, 5 assets, 2009-09-28 to 2026-10-30 (22,028 asset-days, 36 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: fixed: one estimation on all data before the cutoff; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: EWMA lambda 0.94, ARCH order 5, GARCH errors normal, HAR estimation pooled with lags 1/5/22
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

9,778 forecasts on 2,026 days (2019-01-02 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|
|      1 | AMEM      |  0.1683 |           -2.3274 |   -1.5135 |       0.1302 |               1 |              1 |        1.0000 | yes      |       0.3149 |
|      2 | MEM       |  0.1700 |           -1.3513 |   -0.8989 |       0.3687 |               0 |              1 |        0.1495 | yes      |       0.3192 |
|      3 | logSHAR   |  0.1714 |           -0.5628 |   -1.4994 |       0.1338 |               1 |              0 |        0.3225 | yes      |       0.2894 |
|      4 | logHAR    |  0.1723 |            0.0000 |           |              |                 |                |        0.3225 | yes      |       0.2930 |
|      5 | logHAR-MJ |  0.1731 |            0.4208 |    1.2856 |       0.1986 |               0 |              1 |        0.1195 | yes      |       0.2912 |
|      6 | EWMA-RV   |  0.1759 |            2.0617 |    1.7340 |       0.0829 |               0 |              2 |        0.0000 |          |       0.3191 |
|      7 | HAR       |  0.1882 |            9.2178 |    6.1151 |       0.0000 |               0 |              3 |        0.0000 |          |       0.3958 |
|      8 | GARCH     |  0.2080 |           20.6583 |    6.8363 |       0.0000 |               0 |              4 |        0.0000 |          |       0.4083 |
|      9 | GJR-GARCH |  0.2197 |           27.4588 |    5.0350 |       0.0000 |               0 |              4 |        0.0000 |          |       0.4066 |
|     10 | EGARCH    |  0.2253 |           30.7360 |    3.7534 |       0.0002 |               0 |              4 |        0.0035 |          |       0.3744 |
|     11 | EWMA      |  0.2315 |           34.3057 |    7.9744 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4447 |
|     12 | HARQ      |  0.2404 |           39.5080 |    1.3389 |       0.1806 |               0 |              2 |        0.3225 | yes      |       0.3700 |
|     13 | ARCH      |  0.2580 |           49.7179 |   12.9536 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4710 |
|     14 | RW        |  0.2582 |           49.7945 |    9.9013 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3888 |

## Test from 2021-01-01

7,289 forecasts on 1,509 days (2021-01-04 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|
|      1 | AMEM      |  0.1685 |           -0.8158 |   -0.4578 |       0.6471 |               0 |              1 |        1.0000 | yes      |       0.3147 |
|      2 | logSHAR   |  0.1689 |           -0.5796 |   -1.3533 |       0.1760 |               1 |              0 |        0.9475 | yes      |       0.2843 |
|      3 | MEM       |  0.1695 |           -0.2028 |   -0.0122 |       0.9902 |               0 |              1 |        0.5830 | yes      |       0.3170 |
|      4 | logHAR    |  0.1699 |            0.0000 |           |              |                 |                |        0.4635 | yes      |       0.2878 |
|      5 | logHAR-MJ |  0.1706 |            0.4111 |    1.1820 |       0.2372 |               0 |              1 |        0.1180 | yes      |       0.2860 |
|      6 | EWMA-RV   |  0.1751 |            3.0856 |    2.4156 |       0.0157 |               0 |              2 |        0.0000 |          |       0.3163 |
|      7 | GARCH     |  0.1936 |           13.9570 |    5.5370 |       0.0000 |               0 |              4 |        0.0000 |          |       0.3801 |
|      8 | HAR       |  0.1966 |           15.7477 |    7.9449 |       0.0000 |               0 |              3 |        0.0000 |          |       0.4233 |
|      9 | GJR-GARCH |  0.1990 |           17.1658 |    4.0446 |       0.0001 |               0 |              4 |        0.0015 |          |       0.3687 |
|     10 | EGARCH    |  0.2068 |           21.7578 |    2.9035 |       0.0037 |               0 |              3 |        0.0345 |          |       0.3665 |
|     11 | EWMA      |  0.2143 |           26.1786 |    7.1521 |       0.0000 |               0 |              4 |        0.0000 |          |       0.4082 |
|     12 | ARCH      |  0.2496 |           46.9479 |   11.8032 |       0.0000 |               0 |              5 |        0.0000 |          |       0.4653 |
|     13 | HARQ      |  0.2601 |           53.1180 |    1.3113 |       0.1898 |               0 |              1 |        0.4635 | yes      |       0.3548 |
|     14 | RW        |  0.2601 |           53.1465 |    8.5957 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3879 |

## How to read the table

- **QLIKE** is the loss to minimize (0 = perfect forecast). Models are ranked on it.
- **DM stat / p-value**: Diebold-Mariano test against logHAR on the daily loss averaged across assets. A negative statistic means the model beats the benchmark; the difference is significant at 5 % when the p-value is below 0.05.
- **assets better / worse**: number of assets (out of 5) where the model is significantly better / worse than the benchmark at 5 % when the test is run asset by asset.
- **in MCS**: the model belongs to the Model Confidence Set, the smallest group of models that contains the best one with the stated confidence. Models outside it are significantly worse than the best.
- **MSE log RV** is the criterion of the DL and foundation-model branches, on the same asset-days when the dataset is stocks.

## Figures

![QLIKE ranking](figures/qlike_ranking.png)

![Diebold-Mariano matrix](figures/dm_matrix.png)

![Forecast path](figures/forecast_path.png)
