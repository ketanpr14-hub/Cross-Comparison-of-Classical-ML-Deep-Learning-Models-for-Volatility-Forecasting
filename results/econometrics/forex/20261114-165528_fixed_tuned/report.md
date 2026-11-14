# Econometric volatility forecasts: forex

Run `20261114-165528_fixed_tuned` (2026-11-14T16:55:32, 79.3 s, commit f59e035).

## Settings

- **Dataset**: forex, 5 assets, 2009-09-25 to 2026-10-30 (22,178 asset-days, 0 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: fixed: one estimation on all data before the cutoff; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: selected by QLIKE on 2 validation year(s) before each cutoff, then the model is re-estimated on the whole training period (values in the tables below)
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

10,160 forecasts on 2,032 days (2019-01-02 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1511 |           -1.4820 |   -2.1943 |       0.0282 |               2 |              0 |        1.0000 | yes      |       0.2491 | 9.574e-10 |
|      2 | logSHAR   |  0.1527 |           -0.4789 |   -2.2941 |       0.0218 |               1 |              0 |        0.2280 | yes      |       0.2506 | 9.784e-10 |
|      3 | logHAR    |  0.1534 |            0.0000 |           |              |                 |                |        0.1425 | yes      |       0.2520 | 9.868e-10 |
|      4 | AMEM      |  0.1552 |            1.1674 |    0.6022 |       0.5470 |               0 |              1 |        0.2470 | yes      |       0.3054 | 9.948e-10 |
|      5 | MEM       |  0.1552 |            1.1833 |    0.6170 |       0.5372 |               0 |              2 |        0.2470 | yes      |       0.3055 | 1.006e-09 |
|      6 | HARQ      |  0.1578 |            2.8237 |    1.1540 |       0.2485 |               0 |              3 |        0.2210 | yes      |       0.3212 | 1.007e-09 |
|      7 | HAR       |  0.1601 |            4.3777 |    1.9168 |       0.0553 |               0 |              3 |        0.0370 |          |       0.3300 | 9.945e-10 |
|      8 | EWMA-RV   |  0.1631 |            6.2762 |    3.3908 |       0.0007 |               0 |              4 |        0.0050 |          |       0.2930 | 1.031e-09 |
|      9 | GJR-GARCH |  0.2020 |           31.6493 |    4.4757 |       0.0000 |               0 |              5 |        0.0010 |          |       0.3961 | 1.179e-09 |
|     10 | GARCH     |  0.2021 |           31.7228 |    4.0353 |       0.0001 |               0 |              4 |        0.0040 |          |       0.3879 | 1.213e-09 |
|     11 | EGARCH    |  0.2022 |           31.7671 |    3.9198 |       0.0001 |               0 |              5 |        0.0050 |          |       0.3722 | 1.189e-09 |
|     12 | EWMA      |  0.2193 |           42.9468 |    3.5407 |       0.0004 |               0 |              5 |        0.0085 |          |       0.3841 | 1.303e-09 |
|     13 | RW        |  0.2198 |           43.2628 |    8.7100 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3459 | 1.506e-09 |
|     14 | ARCH      |  0.2317 |           50.9972 |   11.3424 |       0.0000 |               0 |              5 |        0.0000 |          |       0.5135 | 1.295e-09 |

Hyperparameters used:

| model     | hyperparameters                      |   candidates |   validation QLIKE |   with default values |
|:----------|:-------------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                                 |            1 |                    |                       |
| EWMA      | lambda = 0.97                        |            6 |             0.1349 |                0.1429 |
| EWMA-RV   | none                                 |            1 |                    |                       |
| ARCH      | p = 10, dist = t                     |           10 |             0.1501 |                0.1704 |
| GARCH     | p = 1, q = 2, dist = t               |            8 |             0.1316 |                0.1333 |
| GJR-GARCH | p = 1, q = 2, dist = normal          |            4 |             0.1372 |                0.1381 |
| EGARCH    | p = 1, q = 2, dist = normal          |            4 |             0.1394 |                0.1405 |
| HAR       | pooling = pooled, estimator = wls    |            4 |             0.1200 |                0.1261 |
| HARQ      | pooling = per_asset, estimator = wls |            4 |             0.1206 |                0.1250 |
| logHAR    | pooling = pooled                     |            2 |             0.1160 |                0.1160 |
| logHAR-MJ | pooling = per_asset                  |            2 |             0.1128 |                0.1131 |
| logSHAR   | pooling = pooled                     |            2 |             0.1156 |                0.1156 |
| MEM       | p = 1, q = 2                         |            4 |             0.1199 |                0.1221 |
| AMEM      | p = 2, q = 2                         |            4 |             0.1210 |                0.1228 |

## Test from 2021-01-01

7,565 forecasts on 1,513 days (2021-01-04 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1555 |           -1.3450 |   -1.9019 |       0.0572 |               2 |              0 |        1.0000 | yes      |       0.2489 | 6.931e-10 |
|      2 | HAR       |  0.1566 |           -0.6601 |   -0.2678 |       0.7888 |               0 |              4 |        0.7620 | yes      |       0.2992 | 7.518e-10 |
|      3 | logSHAR   |  0.1569 |           -0.4868 |   -2.1065 |       0.0352 |               2 |              0 |        0.4750 | yes      |       0.2500 | 6.971e-10 |
|      4 | logHAR    |  0.1576 |            0.0000 |           |              |                 |                |        0.2325 | yes      |       0.2515 | 7.015e-10 |
|      5 | MEM       |  0.1579 |            0.1697 |    0.0949 |       0.9244 |               0 |              4 |        0.6205 | yes      |       0.2992 | 7.497e-10 |
|      6 | AMEM      |  0.1589 |            0.8298 |    0.4984 |       0.6182 |               0 |              2 |        0.4750 | yes      |       0.2999 | 7.545e-10 |
|      7 | HARQ      |  0.1598 |            1.4033 |    0.5778 |       0.5634 |               0 |              4 |        0.0015 |          |       0.3097 | 7.365e-10 |
|      8 | EWMA-RV   |  0.1668 |            5.8354 |    3.3766 |       0.0007 |               0 |              4 |        0.0045 |          |       0.2896 | 7.432e-10 |
|      9 | GJR-GARCH |  0.1845 |           17.0641 |    4.0442 |       0.0001 |               0 |              4 |        0.0005 |          |       0.3499 | 7.750e-10 |
|     10 | GARCH     |  0.1847 |           17.1721 |    3.8652 |       0.0001 |               0 |              4 |        0.0005 |          |       0.3533 | 7.836e-10 |
|     11 | EGARCH    |  0.1868 |           18.5169 |    4.8485 |       0.0000 |               0 |              4 |        0.0005 |          |       0.3423 | 7.752e-10 |
|     12 | ARCH      |  0.2166 |           37.4266 |    8.3742 |       0.0000 |               0 |              4 |        0.0000 |          |       0.4541 | 8.682e-10 |
|     13 | EWMA      |  0.2208 |           40.1087 |    6.4031 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3557 | 8.042e-10 |
|     14 | RW        |  0.2293 |           45.4581 |    7.8841 |       0.0000 |               0 |              5 |        0.0000 |          |       0.3522 | 1.060e-09 |

Hyperparameters used:

| model     | hyperparameters                      |   candidates |   validation QLIKE |   with default values |
|:----------|:-------------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                                 |            1 |                    |                       |
| EWMA      | lambda = 0.9                         |            6 |             0.2367 |                0.2408 |
| EWMA-RV   | none                                 |            1 |                    |                       |
| ARCH      | p = 10, dist = t                     |           10 |             0.2448 |                0.2863 |
| GARCH     | p = 1, q = 2, dist = normal          |            8 |             0.2419 |                0.2441 |
| GJR-GARCH | p = 1, q = 2, dist = normal          |            4 |             0.2429 |                0.2490 |
| EGARCH    | p = 1, q = 2, dist = normal          |            4 |             0.2407 |                0.2429 |
| HAR       | pooling = per_asset, estimator = wls |            4 |             0.1557 |                0.1898 |
| HARQ      | pooling = pooled, estimator = wls    |            4 |             0.1537 |                0.1796 |
| logHAR    | pooling = pooled                     |            2 |             0.1419 |                0.1419 |
| logHAR-MJ | pooling = per_asset                  |            2 |             0.1405 |                0.1415 |
| logSHAR   | pooling = pooled                     |            2 |             0.1412 |                0.1412 |
| MEM       | p = 1, q = 2                         |            4 |             0.1476 |                0.1489 |
| AMEM      | p = 2, q = 2                         |            4 |             0.1472 |                0.1480 |

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
