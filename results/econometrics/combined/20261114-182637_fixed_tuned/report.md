# Econometric volatility forecasts: combined

Run `20261114-182637_fixed_tuned` (2026-11-14T18:27:11, 271.8 s, commit 737cb4a).

## Settings

- **Dataset**: combined, 50 assets, 2009-09-25 to 2026-10-30 (163,206 asset-days, 42 masked)
- **Target**: log(rv5) of the next session; QLIKE is scored on rv5 itself
- **Splitting policy**: fixed: one estimation on all data before the cutoff; test cutoffs 2019-01-01, 2021-01-01
- **Hyperparameters**: selected by QLIKE on 2 validation year(s) before each cutoff, then the model is re-estimated on the whole training period (values in the tables below)
- **Tests**: benchmark logHAR; Model Confidence Set at 90 % (2000 stationary-bootstrap draws, mean block 10 days)

## Test from 2019-01-01

98,543 forecasts on 2,032 days (2019-01-02 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1769 |           -2.5244 |   -2.1439 |       0.0320 |              20 |              2 |        1.0000 | yes      |       0.2814 | 2.432e-07 |
|      2 | logSHAR   |  0.1810 |           -0.2642 |   -1.1758 |       0.2397 |              11 |              1 |        0.0730 |          |       0.2868 | 2.528e-07 |
|      3 | logHAR    |  0.1815 |            0.0000 |           |              |                 |                |        0.0730 |          |       0.2875 | 2.540e-07 |
|      4 | MEM       |  0.1879 |            3.5621 |    1.7645 |       0.0777 |               0 |             17 |        0.0005 |          |       0.3269 | 2.778e-07 |
|      5 | AMEM      |  0.1892 |            4.2767 |    2.0223 |       0.0431 |               1 |             18 |        0.0005 |          |       0.3281 | 2.813e-07 |
|      6 | HAR       |  0.1907 |            5.0883 |    3.5475 |       0.0004 |               0 |             16 |        0.0000 |          |       0.3361 | 2.682e-07 |
|      7 | EWMA-RV   |  0.1979 |            9.0750 |    4.5585 |       0.0000 |               0 |             34 |        0.0000 |          |       0.3282 | 2.632e-07 |
|      8 | HARQ      |  0.2111 |           16.3477 |    2.5031 |       0.0123 |               0 |             13 |        0.0195 |          |       0.3387 | 2.457e-07 |
|      9 | RW        |  0.2466 |           35.8874 |   13.9847 |       0.0000 |               0 |             50 |        0.0000 |          |       0.3822 | 3.609e-07 |
|     10 | GARCH     |  0.2591 |           42.8104 |    5.2739 |       0.0000 |               0 |             49 |        0.0005 |          |       0.4242 | 3.365e-07 |
|     11 | EWMA      |  0.2664 |           46.7960 |    5.8720 |       0.0000 |               0 |             50 |        0.0005 |          |       0.4354 | 3.412e-07 |
|     12 | GJR-GARCH |  0.2671 |           47.1685 |    5.3648 |       0.0000 |               0 |             50 |        0.0005 |          |       0.4258 | 3.560e-07 |
|     13 | EGARCH    |  0.2705 |           49.0664 |    4.2230 |       0.0000 |               0 |             48 |        0.0040 |          |       0.4171 | 5.540e-07 |
|     14 | ARCH      |  0.2889 |           59.2115 |    7.7292 |       0.0000 |               0 |             50 |        0.0000 |          |       0.4653 | 4.061e-07 |

QLIKE by asset class (number of forecasts in brackets):

| model     |   forex (10,160) |   futures (9,778) |   stocks (78,605) |
|:----------|-----------------:|------------------:|------------------:|
| logHAR-MJ |           0.1511 |            0.1695 |            0.1811 |
| logSHAR   |           0.1509 |            0.1683 |            0.1864 |
| logHAR    |           0.1517 |            0.1698 |            0.1867 |
| MEM       |           0.1552 |            0.1692 |            0.1945 |
| AMEM      |           0.1552 |            0.1674 |            0.1963 |
| HAR       |           0.1903 |            0.1742 |            0.1928 |
| EWMA-RV   |           0.1631 |            0.1759 |            0.2052 |
| HARQ      |           0.1863 |            0.3550 |            0.1964 |
| RW        |           0.2198 |            0.2582 |            0.2486 |
| GARCH     |           0.2021 |            0.2115 |            0.2724 |
| EWMA      |           0.2104 |            0.2315 |            0.2780 |
| GJR-GARCH |           0.2036 |            0.2196 |            0.2812 |
| EGARCH    |           0.2041 |            0.2211 |            0.2852 |
| ARCH      |           0.2193 |            0.2301 |            0.3052 |

Hyperparameters used:

| model     | hyperparameters                   |   candidates |   validation QLIKE |   with default values |
|:----------|:----------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                              |            1 |                    |                       |
| EWMA      | lambda = 0.94                     |            6 |             0.2629 |                0.2629 |
| EWMA-RV   | none                              |            1 |                    |                       |
| ARCH      | p = 22, dist = t                  |           10 |             0.2595 |                0.2907 |
| GARCH     | p = 1, q = 2, dist = t            |            8 |             0.2503 |                0.2563 |
| GJR-GARCH | p = 1, q = 2, dist = t            |            4 |             0.2464 |                0.2548 |
| EGARCH    | p = 1, q = 2, dist = t            |            4 |           117.7133 |             8821.3259 |
| HAR       | pooling = pooled, estimator = wls |            4 |             0.1834 |                0.2139 |
| HARQ      | pooling = pooled, estimator = wls |            4 |             0.1863 |                0.1981 |
| logHAR    | pooling = per_asset               |            2 |             0.1792 |                0.1804 |
| logHAR-MJ | pooling = per_asset               |            2 |             0.1712 |                0.1765 |
| logSHAR   | pooling = per_asset               |            2 |             0.1788 |                0.1803 |
| MEM       | p = 2, q = 2                      |            4 |             0.1850 |                0.1877 |
| AMEM      | p = 2, q = 2                      |            4 |             0.1859 |                0.1883 |

## Test from 2021-01-01

73,259 forecasts on 1,513 days (2021-01-04 to 2026-10-29), 1 estimation(s).

|   rank | model     |   QLIKE |   vs benchmark, % |   DM stat |   DM p-value |   assets better |   assets worse |   MCS p-value | in MCS   |   MSE log RV |    MSE RV |
|-------:|:----------|--------:|------------------:|----------:|-------------:|----------------:|---------------:|--------------:|:---------|-------------:|----------:|
|      1 | logHAR-MJ |  0.1736 |           -2.6506 |   -3.7987 |       0.0001 |              24 |              2 |        1.0000 | yes      |       0.2729 | 1.945e-07 |
|      2 | logSHAR   |  0.1777 |           -0.3274 |   -0.3208 |       0.7484 |              12 |              0 |        0.0000 |          |       0.2783 | 1.949e-07 |
|      3 | logHAR    |  0.1783 |            0.0000 |           |              |                 |                |        0.0000 |          |       0.2798 | 1.961e-07 |
|      4 | HAR       |  0.1827 |            2.4410 |    2.0574 |       0.0396 |               0 |             25 |        0.0000 |          |       0.3132 | 2.277e-07 |
|      5 | MEM       |  0.1835 |            2.9087 |    2.0915 |       0.0365 |               0 |             29 |        0.0000 |          |       0.3216 | 2.550e-07 |
|      6 | AMEM      |  0.1839 |            3.1009 |    2.1112 |       0.0348 |               1 |             29 |        0.0000 |          |       0.3213 | 2.612e-07 |
|      7 | EWMA-RV   |  0.1957 |            9.7647 |    8.2001 |       0.0000 |               0 |             43 |        0.0000 |          |       0.3204 | 2.313e-07 |
|      8 | HARQ      |  0.2171 |           21.7603 |    5.3734 |       0.0000 |               0 |             28 |        0.0000 |          |       0.3971 | 2.037e-07 |
|      9 | EGARCH    |  0.2255 |           26.4406 |   10.1076 |       0.0000 |               0 |             45 |        0.0000 |          |       0.3819 | 2.159e-07 |
|     10 | GARCH     |  0.2327 |           30.4927 |   10.3681 |       0.0000 |               0 |             47 |        0.0000 |          |       0.3965 | 2.191e-07 |
|     11 | GJR-GARCH |  0.2336 |           31.0022 |   11.9936 |       0.0000 |               0 |             45 |        0.0000 |          |       0.3923 | 2.226e-07 |
|     12 | RW        |  0.2470 |           38.5099 |   14.6909 |       0.0000 |               0 |             50 |        0.0000 |          |       0.3793 | 3.306e-07 |
|     13 | ARCH      |  0.2543 |           42.5888 |   15.4012 |       0.0000 |               0 |             47 |        0.0000 |          |       0.4309 | 2.287e-07 |
|     14 | EWMA      |  0.2617 |           46.7446 |   13.7699 |       0.0000 |               0 |             49 |        0.0000 |          |       0.4181 | 2.232e-07 |

QLIKE by asset class (number of forecasts in brackets):

| model     |   forex (7,565) |   futures (7,289) |   stocks (58,405) |
|:----------|----------------:|------------------:|------------------:|
| logHAR-MJ |          0.1555 |            0.1680 |            0.1766 |
| logSHAR   |          0.1584 |            0.1703 |            0.1812 |
| logHAR    |          0.1590 |            0.1707 |            0.1818 |
| HAR       |          0.1566 |            0.1686 |            0.1878 |
| MEM       |          0.1579 |            0.1688 |            0.1887 |
| AMEM      |          0.1589 |            0.1671 |            0.1892 |
| EWMA-RV   |          0.1668 |            0.1751 |            0.2021 |
| HARQ      |          0.3492 |            0.2624 |            0.1944 |
| EGARCH    |          0.1854 |            0.2056 |            0.2332 |
| GARCH     |          0.1871 |            0.1966 |            0.2431 |
| GJR-GARCH |          0.1847 |            0.1985 |            0.2443 |
| RW        |          0.2293 |            0.2601 |            0.2477 |
| ARCH      |          0.2019 |            0.2139 |            0.2661 |
| EWMA      |          0.2208 |            0.2164 |            0.2726 |

Hyperparameters used:

| model     | hyperparameters                      |   candidates |   validation QLIKE |   with default values |
|:----------|:-------------------------------------|-------------:|-------------------:|----------------------:|
| RW        | none                                 |            1 |                    |                       |
| EWMA      | lambda = 0.9                         |            6 |             0.3201 |                0.3393 |
| EWMA-RV   | none                                 |            1 |                    |                       |
| ARCH      | p = 22, dist = t                     |           10 |             0.3469 |                0.4189 |
| GARCH     | p = 1, q = 1, dist = t               |            8 |             0.3227 |                0.3324 |
| GJR-GARCH | p = 1, q = 2, dist = t               |            4 |             0.3383 |                0.3527 |
| EGARCH    | p = 1, q = 2, dist = t               |            4 |             0.3635 |                0.3881 |
| HAR       | pooling = per_asset, estimator = wls |            4 |             0.1906 |                0.2356 |
| HARQ      | pooling = pooled, estimator = ols    |            4 |             0.2162 |                0.2162 |
| logHAR    | pooling = pooled                     |            2 |             0.1944 |                0.1944 |
| logHAR-MJ | pooling = per_asset                  |            2 |             0.1867 |                0.1898 |
| logSHAR   | pooling = pooled                     |            2 |             0.1944 |                0.1944 |
| MEM       | p = 1, q = 2                         |            4 |             0.1901 |                0.1915 |
| AMEM      | p = 1, q = 2                         |            4 |             0.1913 |                0.1927 |

## How to read the table

- **QLIKE** is the loss to minimize (0 = perfect forecast). Models are ranked on it.
- **DM stat / p-value**: Diebold-Mariano test against logHAR on the daily loss averaged across assets. A negative statistic means the model beats the benchmark; the difference is significant at 5 % when the p-value is below 0.05.
- **assets better / worse**: number of assets (out of 50) where the model is significantly better / worse than the benchmark at 5 % when the test is run asset by asset.
- **in MCS**: the model belongs to the Model Confidence Set, the smallest group of models that contains the best one with the stated confidence. Models outside it are significantly worse than the best.
- **MSE log RV** is the mean squared error on log(rv5), the criterion of the DL and foundation-model branches, on the same asset-days when the dataset is stocks.
- **MSE RV** is the mean squared error on rv5 itself. A handful of crisis days carry most of it, so it is much noisier than QLIKE.

## Figures

![QLIKE ranking](figures/qlike_ranking.png)

![Diebold-Mariano matrix](figures/dm_matrix.png)

![Forecast path](figures/forecast_path.png)
