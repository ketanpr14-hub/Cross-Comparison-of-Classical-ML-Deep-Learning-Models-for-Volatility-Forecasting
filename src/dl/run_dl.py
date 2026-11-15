"""
MLP and LSTM against logHAR-MJ, on the team protocol. Meant to run on the DCE cluster.

Run from the project root:
    python src/dl/run_dl.py --quick                       # 3 stocks, 1 seed, 3 epochs: checks the code
    python src/dl/run_dl.py                               # fixed split (team protocol)
    python src/dl/run_dl.py --split expanding             # walk-forward: refit every year
    python src/dl/run_dl.py --split rolling               # walk-forward: last 1000 days only
    python src/dl/run_dl.py --dataset forex --models MLP LSTM22

What it does, in order:
  1. load the data with Jean's build_panel (same cleaning, features and test rows as the team)
  2. cut the test period into folds: 1 fold (fixed) or one per year (expanding / rolling)
  3. for each fold:
       - baselines: tomorrow = today, logHAR-MJ pooled (one regression), logHAR-MJ per stock (Jean's)
       - MLP and LSTM: train on the fit years, stop early on the last year before the fold
         (validation), average 3 seeds
       - variance forecast = exp(log forecast + sigma2 / 2), sigma2 = variance of the
         validation errors (same rule for every model)
  4. score every model on the same rows: QLIKE, MSE log RV, MSE RV, DM test vs pooled HAR

Output, in results/dl/:
    <dataset>_<split>.csv            the team table (one line per model and cutoff)
    <dataset>_<split>_by_year.csv    the same scores year by year (to see where a model wins)
    forecasts_<dataset>_<split>.csv.gz   every forecast (large, not committed)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))                          # so we can import Jean's code
from econometrics.config import CUTOFFS, RunConfig              # noqa: E402
from econometrics.data import build_panel                       # noqa: E402
from econometrics.evaluation import dm_test, qlike              # noqa: E402
from econometrics.models import make_folds, regression_model   # noqa: E402

FEATURES = ["x_day", "x_week", "x_month", "m_day", "m_week", "jump_share"]   # logHAR-MJ inputs
CHANNELS = ["y", "m_day", "jump_share"]                                      # what the LSTM reads, day by day

# Fixed hyperparameters, chosen before looking at any result (no grid search).
MLP_HIDDEN = [16, 16]
LSTM_HIDDEN = 32
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 512
PATIENCE = 5
SEEDS, MAX_EPOCHS = [0, 1, 2], 30

# name used on the command line -> (kind, days of memory, name in the tables)
MODELS = {"MLP": ("mlp", None, "MLP [16,16]"),
          "LSTM22": ("lstm", 22, "LSTM 22d"),
          "LSTM66": ("lstm", 66, "LSTM 66d")}
HAR = "logHAR-MJ (pooled)"


# ---------------------------------------------------------------- 1. data
def load_data(dataset, quick):
    panel, _ = build_panel(RunConfig(dataset=dataset, quick=quick))
    panel = panel.reset_index(drop=True)
    # The rows logHAR-MJ can use: all features and a target. On stocks this gives the
    # 78,605 / 58,405 test rows of the protocol.
    panel["eligible"] = panel[FEATURES + ["target", "target_rv"]].notna().all(axis=1)
    return panel


def make_windows(panel, seq):
    """For each eligible row: the last `seq` days of CHANNELS of the same asset (oldest first)."""
    windows = np.full((len(panel), seq, len(CHANNELS)), np.nan, dtype=np.float32)
    for _, g in panel.groupby("symbol"):                          # assets never mix
        values = g[CHANNELS].ffill().bfill().to_numpy(dtype=np.float32)   # a masked day takes the previous value
        padded = np.concatenate([np.repeat(values[:1], seq - 1, axis=0), values])   # first rows repeat day 1
        windows[g.index.to_numpy()] = sliding_window_view(padded, (seq, len(CHANNELS)))[:, 0]
    X = windows[panel["eligible"].to_numpy()]
    el = panel[panel["eligible"]]
    # checks: the windows line up with the HAR features
    assert not np.isnan(X).any()
    assert np.allclose(X[:, -1, 0], el["x_day"], atol=1e-4)
    if seq >= 22:
        assert np.allclose(X[:, -22:, 0].mean(1), el["x_month"], atol=1e-4)
    return X


# ---------------------------------------------------------------- 2. models
def make_mlp(n_in):
    """Same structure as stocks_har_mlp.py: Linear + ReLU layers, linear output."""
    layers, size = [], n_in
    for h in MLP_HIDDEN:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers, nn.Flatten(0))        # output shape (batch,)


class LSTM(nn.Module):
    """Reads the days in order and predicts from its memory after the last day."""

    def __init__(self, n_channels):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, LSTM_HIDDEN, batch_first=True)
        self.out = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        states, _ = self.lstm(x)                       # memory after each day
        return self.out(states[:, -1]).squeeze(-1)     # forecast from the last memory


# ---------------------------------------------------------------- 3. training
def predict(model, X, batch=8192):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)]).numpy()


def train_early_stop(model, X_fit, y_fit, X_val, y_val, seed):
    """Mini-batch Adam; keep the weights of the epoch with the best VALIDATION loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_fit))              # new random order of the rows each epoch
        for i in range(0, len(X_fit), BATCH):
            rows = order[i:i + BATCH]
            loss = nn.functional.mse_loss(model(X_fit[rows]), y_fit[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = nn.functional.mse_loss(torch.from_numpy(predict(model, X_val)), y_val).item()
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)                   # back to the best epoch
    return best_epoch


def fit_forecast(kind, X, y, fit, val, test):
    """
    X: inputs of all eligible rows, y: their target (log variance). fit / val / test: row numbers.
    Returns the log forecast of the test rows, and sigma2 = variance of the validation errors.
    """
    # standardize with the fit rows only (never validation or test)
    flat = X[fit].reshape(-1, X.shape[-1])
    Xn = ((X - flat.mean(0)) / flat.std(0)).astype(np.float32)
    y_mu, y_sd = y[fit].mean(), y[fit].std()
    yn = torch.tensor((y - y_mu) / y_sd, dtype=torch.float32)
    Xn = torch.from_numpy(Xn)

    val_preds, test_preds, epochs = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = make_mlp(X.shape[-1]) if kind == "mlp" else LSTM(X.shape[-1])
        epochs.append(train_early_stop(model, Xn[fit], yn[fit], Xn[val], yn[val], seed))
        val_preds.append(predict(model, Xn[val]))
        test_preds.append(predict(model, Xn[test]))
    unscale = lambda p: np.mean(p, axis=0) * y_sd + y_mu          # average the seeds, back to log RV
    val_pred, test_pred = unscale(val_preds), unscale(test_preds)
    return test_pred, np.var(y[val] - val_pred), epochs


# ---------------------------------------------------------------- 4. one cutoff
def run_cutoff(panel, inputs, cutoff, args):
    cfg = RunConfig(dataset=args.dataset, split_policy=args.split, window=args.window)
    folds = make_folds(panel["date"], cutoff, cfg)       # Jean's folds: same refit dates as his runs

    el = panel[panel["eligible"]].reset_index(drop=True)
    dates, y = el["date"], el["target"].to_numpy()
    in_test = (dates >= cutoff).to_numpy()

    # baseline 1: Jean's logHAR-MJ, one regression per stock, with his own conversion to variance
    h_jean, logf_jean, _ = regression_model(panel, folds, "logHAR-MJ", {"pooling": "per_asset"}, cfg)
    elig = panel["eligible"].to_numpy()
    jean = "logHAR-MJ (Jean, per asset)"
    log_fc = {"RW": el["x_day"].to_numpy(), jean: logf_jean.to_numpy()[elig], HAR: np.full(len(el), np.nan)}
    var_fc = {"RW": np.exp(el["x_day"].to_numpy()), jean: h_jean.to_numpy()[elig], HAR: np.full(len(el), np.nan)}
    names = {m: MODELS[m][2] for m in args.models}
    for n in names.values():
        log_fc[n], var_fc[n] = np.full(len(el), np.nan), np.full(len(el), np.nan)

    for k, fold in enumerate(folds):
        t0 = time.time()
        val_start = fold.test_start - pd.DateOffset(years=1)
        train = ((dates >= fold.train_start) & (dates < fold.test_start)).to_numpy()   # fit + validation years
        fit = np.flatnonzero((dates >= fold.train_start) & (dates < val_start))
        val = np.flatnonzero((dates >= val_start) & (dates < fold.test_start))
        test = np.flatnonzero((dates >= fold.test_start) & (dates < fold.test_end))

        # baseline 2: pooled logHAR-MJ. Final regression on all training years; sigma2 from
        # a regression on the fit years, measured on the validation year (like the networks).
        har = LinearRegression().fit(el.loc[train, FEATURES], y[train])
        har_fit = LinearRegression().fit(el.loc[fit, FEATURES], y[fit])
        sigma2 = np.var(y[val] - har_fit.predict(el.loc[val, FEATURES]))
        log_fc[HAR][test] = har.predict(el.loc[test, FEATURES])
        var_fc[HAR][test] = np.exp(log_fc[HAR][test] + sigma2 / 2)

        msg = []
        for m, name in names.items():
            kind = MODELS[m][0]
            pred, sigma2, epochs = fit_forecast(kind, inputs[m], y, fit, val, test)
            log_fc[name][test] = pred
            var_fc[name][test] = np.exp(pred + sigma2 / 2)
            msg.append(f"{name}: epochs {epochs}")
        print(f"  fold {k}: test from {fold.test_start.date()} | fit {len(fit):,} val {len(val):,} "
              f"test {len(test):,} rows | {' | '.join(msg)} | {time.time() - t0:.0f}s", flush=True)

    # ---- score every model on the same test rows
    rv, logrv = el["target_rv"].to_numpy()[in_test], y[in_test]
    day, year = dates[in_test].to_numpy(), dates[in_test].dt.year.to_numpy()
    losses = {}
    for name in log_fc:
        assert not np.isnan(var_fc[name][in_test]).any(), f"{name} has missing forecasts"
        losses[name] = qlike(rv, var_fc[name][in_test])
    daily_har = pd.Series(losses[HAR]).groupby(day).mean()

    table, by_year = [], []
    for name in log_fc:
        h, f = var_fc[name][in_test], log_fc[name][in_test]
        # DM test on the daily average loss (stocks on the same day are not independent)
        stat, p = dm_test(pd.Series(losses[name]).groupby(day).mean(), daily_har) if name != HAR else (np.nan, np.nan)
        table.append({"dataset": args.dataset, "split": args.split, "cutoff": cutoff, "model": name,
                      "n_forecasts": int(in_test.sum()), "QLIKE": losses[name].mean(),
                      "MSE log RV": np.mean((logrv - f) ** 2), "MSE RV": np.mean((rv - h) ** 2),
                      "DM stat vs HAR": stat, "DM p-value vs HAR": p})
        yearly = pd.DataFrame({"year": year, "se": (logrv - f) ** 2, "ql": losses[name]}).groupby("year").agg(
            n_forecasts=("se", "size"), **{"MSE log RV": ("se", "mean"), "QLIKE": ("ql", "mean")}).reset_index()
        by_year.append(yearly.assign(dataset=args.dataset, split=args.split, cutoff=cutoff, model=name))

    table = pd.DataFrame(table)
    har_row = table[table["model"] == HAR].iloc[0]
    table["MSE vs HAR (%)"] = 100 * (table["MSE log RV"] / har_row["MSE log RV"] - 1)
    table["QLIKE vs HAR (%)"] = 100 * (table["QLIKE"] / har_row["QLIKE"] - 1)

    forecasts = el.loc[in_test, ["date", "symbol", "target", "target_rv"]].assign(cutoff=cutoff)
    for name in log_fc:
        forecasts[f"{name} log"], forecasts[f"{name} var"] = log_fc[name][in_test], var_fc[name][in_test]
    return table, pd.concat(by_year, ignore_index=True), forecasts


# ---------------------------------------------------------------- 5. main
def main():
    global SEEDS, MAX_EPOCHS
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="stocks", choices=["stocks", "forex", "futures"])
    p.add_argument("--split", default="fixed", choices=["fixed", "expanding", "rolling"])
    p.add_argument("--window", type=int, default=1000, help="training days for --split rolling")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--quick", action="store_true", help="3 assets, 1 seed, 3 epochs (numbers mean nothing)")
    args = p.parse_args()
    if args.quick:
        SEEDS, MAX_EPOCHS = [0], 3
    tag = "_quick" if args.quick else ""

    t0 = time.time()
    panel = load_data(args.dataset, args.quick)
    print(f"{args.dataset}: {panel['eligible'].sum():,} usable rows, {panel['symbol'].nunique()} assets, "
          f"split {args.split}, models {args.models}, seeds {SEEDS}", flush=True)
    inputs = {}
    for m in args.models:
        kind, seq, _ = MODELS[m]
        inputs[m] = panel.loc[panel["eligible"], FEATURES].to_numpy() if kind == "mlp" else make_windows(panel, seq)

    results = []
    for cutoff in CUTOFFS:
        print(f"\ncutoff {cutoff}", flush=True)
        results.append(run_cutoff(panel, inputs, cutoff, args))
    table = pd.concat([r[0] for r in results], ignore_index=True)
    by_year = pd.concat([r[1] for r in results], ignore_index=True)
    forecasts = pd.concat([r[2] for r in results], ignore_index=True)

    shown = ["cutoff", "model", "n_forecasts", "QLIKE", "MSE log RV", "MSE vs HAR (%)", "QLIKE vs HAR (%)", "DM p-value vs HAR"]
    print("\n" + table[shown].to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    out = ROOT / "results" / "dl"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.split}{tag}"
    table.to_csv(out / f"{stem}.csv", index=False)
    by_year.to_csv(out / f"{stem}_by_year.csv", index=False)
    forecasts.to_csv(out / f"forecasts_{stem}.csv.gz", index=False)
    print(f"\nsaved {stem}.csv, {stem}_by_year.csv, forecasts_{stem}.csv.gz in results/dl/   "
          f"total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
