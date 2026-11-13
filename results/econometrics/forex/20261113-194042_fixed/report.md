# Econometric volatility forecasts: forex

Run `20261113-194042_fixed` (2026-11-13T19:40:43, 11.3 s, commit 80dc556).

## Settings

- **Dataset**: forex, 5 assets, 2009-09-25 to 2026-10-30 (22,178 asset-days, 0 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: fixed: one estimation on all data before the cutoff; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: EWMA lambda 0.94, ARCH order 5, GARCH errors normal, HAR estimation pooled with lags 1/5/22
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

10,160 forecasts on 2,032 days (2019-01-02 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|
|      1 | logSHAR   |  0.1527 |           -0.4789 |   -2.2941 |       0.0218 |               1 |              0 |        1.0000 | yes      |       0.2506 |
|      2 | logHAR-MJ |  0.1531 |           -0.1814 |   -0.3468 |       0.7288 |               1 |              0 |        0.5440 | yes      |       0.2501 |
|      3 | logHAR    |  0.1534 |            0.0000 |           |              |                 |                |        0.1160 | yes      |       0.2520 |
|      4 | MEM       |  0.1568 |            2.1994 |    1.1246 |       0.2608 |               0 |              2 |        0.3880 | yes      |       0.3097 |
|      5 | AMEM      |  0.1568 |            2.2102 |    1.1207 |       0.2624 |               0 |              2 |        0.3880 | yes      |       0.3096 |
|      6 | EWMA-RV   |  0.1631 |            6.2762 |    3.3908 |       0.0007 |               0 |              4 |        0.0025 |          |       0.2930 |
|      7 | HARQ      |  0.1767 |           15.1540 |    5.2388 |       0.0000 |               0 |              4 |        0.0000 |          |       0.3902 |
|      8 | HAR       |  0.1811 |           18.0239 |    5.8874 |       0.0000 |               0 |              4 |        0.0000 |          |       0.4053 |
|      9 | GARCH     |  0.2018 |           31.5267 |    4.4623 |       0.0000 |               0 |              5 |        0.0005 |          |       0.3975 |
|     10 | EGARCH    |  0.2052 |           33.7628 |    4.0280 |       0.0001 |               0 |              5 |        0.0025 |          |       0.3773 |
|     11 | GJR-GARCH |  0.2068 |           34.8237 |    4.4657 |       0.0000 |               0 |              5 |        0.0005 |          |       0.4020 |
|     12 | EWMA      |  0.2104 |           37.1331 |    4.7660 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3588 |
|     13 | RW        |  0.2198 |           43.2628 |    8.7100 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3459 |
|     14 | ARCH      |  0.2653 |           72.9516 |   13.6237 |       0.0000 |               0 |              5 |        0.0000 |          |       0.6400 |

## Test from 2021-01-01

7,565 forecasts on 1,513 days (2021-01-04 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|
|      1 | logSHAR   |  0.1569 |           -0.4868 |   -2.1065 |       0.0352 |               2 |              0 |        1.0000 | yes      |       0.2500 |
|      2 | logHAR-MJ |  0.1573 |           -0.2049 |   -0.3360 |       0.7369 |               1 |              0 |        0.6435 | yes      |       0.2487 |
|      3 | logHAR    |  0.1576 |            0.0000 |           |              |                 |                |        0.1755 | yes      |       0.2515 |
|      4 | MEM       |  0.1600 |            1.4917 |    0.8359 |       0.4032 |               0 |              4 |        0.3755 | yes      |       0.3036 |
|      5 | AMEM      |  0.1614 |            2.3686 |    1.4832 |       0.1380 |               0 |              4 |        0.1755 | yes      |       0.3045 |
|      6 | EWMA-RV   |  0.1668 |            5.8354 |    3.3766 |       0.0007 |               0 |              4 |        0.0065 |          |       0.2896 |
|      7 | HARQ      |  0.1678 |            6.4910 |    2.2574 |       0.0240 |               0 |              4 |        0.0580 |          |       0.3450 |
|      8 | HAR       |  0.1734 |           10.0436 |    3.0870 |       0.0020 |               0 |              3 |        0.0025 |          |       0.3694 |
|      9 | GARCH     |  0.1855 |           17.7047 |    4.0338 |       0.0001 |               0 |              4 |        0.0020 |          |       0.3561 |
|     10 | GJR-GARCH |  0.1856 |           17.7390 |    4.2277 |       0.0000 |               0 |              4 |        0.0010 |          |       0.3528 |
|     11 | EGARCH    |  0.1883 |           19.4684 |    4.9931 |       0.0000 |               0 |              4 |        0.0000 |          |       0.3457 |
|     12 | EWMA      |  0.1998 |           26.7722 |    4.7503 |       0.0000 |               0 |              4 |        0.0005 |          |       0.3375 |
|     13 | RW        |  0.2293 |           45.4581 |    7.8841 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3522 |
|     14 | ARCH      |  0.2383 |           51.2150 |    9.8790 |       0.0000 |               0 |              4 |        0.0000 |          |       0.5338 |

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
