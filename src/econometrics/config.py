"""
Settings shared by the whole econometrics pipeline.

Everything that defines a run (dataset, window, splitting policy, hyperparameters)
lives in `RunConfig`, and is written next to the results of each run.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "econometrics"

DATASETS = {
    name: DATA_DIR / f"realized_variance_{name}_20261030" / f"realized_variance_{name}_20261030.csv"
    for name in ("stocks", "forex", "futures")
}

# Same test cutoffs as the DL and foundation-model branches: train = before, test = after.
CUTOFFS = ("2019-01-01", "2021-01-01")

# Asset-days whose realized measures are wrong. They are set to missing before anything
# is computed from them.
#   stocks : the 6 stock-days found in Quentin's EDA (NYSE opening-auction glitch of
#            2023-01-24, and PG on 2016-09-27). Same list as the other branches.
#   futures: gold on 2009-12-29 has a daily variance of 0.96 (98 % daily vol) for a
#            0.6 % open-to-close move.
BAD_DAYS = {
    "stocks": [("2023-01-24", "MCD"), ("2023-01-24", "NKE"), ("2023-01-24", "V"),
               ("2023-01-24", "WMT"), ("2023-01-24", "XOM"), ("2016-09-27", "PG")],
    "forex": [],
    "futures": [("2009-12-29", "GC")],
}

# Futures only: days where one discontinuity carries more than 90 % of the variance
# (rv5 / bv5 > 10). In this file they are almost all contract rolls (NG around expiry,
# corn at the old-crop/new-crop switch): the "return" is the spread between two
# contracts, not volatility. The rule uses same-day data only, so it is causal.
# It is not applied to stocks, where the same signature is a genuine earnings jump.
JUMP_FILTER = {"stocks": None, "forex": None, "futures": 10.0}

# Realized measures and prices blanked on a masked day.
MEASURES = ["rv5", "bv5", "rq5", "rsp5", "rsn5", "open_price", "close_price"]

ALL_MODELS = [
    "RW", "EWMA", "EWMA-RV", "ARCH", "GARCH", "GJR-GARCH", "EGARCH",
    "HAR", "HARQ", "logHAR", "logHAR-MJ", "logSHAR", "MEM", "AMEM",
]


@dataclass
class RunConfig:
    # ---- data
    dataset: str = "stocks"
    start: str | None = None              # first date kept (None = start of the file)
    end: str | None = None                # last date kept (None = end of the file)
    symbols: list[str] | None = None      # None = every asset of the file
    # ---- splitting policy
    cutoffs: tuple[str, ...] = CUTOFFS
    split_policy: str = "fixed"           # fixed | expanding | rolling
    refit_every: int = 252                # trading days between refits (expanding / rolling)
    window: int = 1000                    # training length in trading days (rolling)
    # ---- models and hyperparameters
    models: list[str] = field(default_factory=lambda: list(ALL_MODELS))
    ewma_lambda: float = 0.94             # RiskMetrics decay of EWMA on squared returns
    arch_p: int = 5                       # order of the ARCH(p) model
    garch_dist: str = "normal"            # normal (QMLE) | t
    har_pooling: str = "pooled"           # pooled (one regression for all assets) | per_asset
    har_lags: tuple[int, int] = (5, 22)   # weekly and monthly windows
    min_train: int = 500                  # minimum observations to estimate a per-asset model
    # ---- testing
    benchmark: str = "logHAR"             # reference model of the Diebold-Mariano columns
    dm_lags: int | None = None            # HAC lags (None = floor(T^(1/3)))
    mcs_size: float = 0.10                # the MCS keeps the best model with 90 % confidence
    mcs_reps: int = 2000
    mcs_block: int = 10                   # mean block length of the stationary bootstrap
    seed: int = 0
    n_jobs: int = -1
    quick: bool = False                   # 3 assets, few bootstrap draws: only to check the code

    def to_dict(self):
        return asdict(self)
