"""
HAR and MLP volatility forecasts on the stocks file (all 40 stocks pooled).

Run from the project root:
    python src/dl/stocks_har_mlp.py            # full run
    python src/dl/stocks_har_mlp.py --quick    # tiny run, just to check that it works

What it does, in order:
  1. load the data, mask the 6 bad stock-days, build the target and the features
  2. for each cutoff (train = before, test = after):
       - baselines: "tomorrow = today" and HAR (linear regression)
       - MLP with early stopping, standardized inputs, several seeds, several architectures
  3. print one table and save it to a CSV
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA = "data/realized_variance_stocks_20261030/realized_variance_stocks_20261030.csv"
CUTOFFS = ["2019-01-01", "2021-01-01"]

# 6 stock-days where rv5 is clearly wrong (found in the EDA). We set rv5 to missing there.
BAD_DAYS = [("2023-01-24", "MCD"), ("2023-01-24", "NKE"), ("2023-01-24", "V"),
            ("2023-01-24", "WMT"), ("2023-01-24", "XOM"), ("2016-09-27", "PG")]

BASE = ["x_day", "x_week", "x_month"]
FEATURE_SETS = {
    "base": BASE,
    "base + market + jump": BASE + ["m_day", "m_week", "jump_share"],
}


# ---------------------------------------------------------------- 1. data
def load_data(path):
    stocks = pd.read_csv(path, parse_dates=["date"])
    df = stocks.sort_values(["symbol", "date"]).copy()

    # mask the bad days BEFORE computing anything from rv5
    for d, s in BAD_DAYS:
        df.loc[(df["date"] == d) & (df["symbol"] == s), "rv5"] = np.nan

    df["y"] = np.log(df["rv5"])                                   # log volatility

    # market-wide features: average log-vol of all stocks on each date
    mkt = df.groupby("date")["y"].mean()
    df["m_day"] = df["date"].map(mkt)
    df["m_week"] = df["date"].map(mkt.rolling(5).mean())

    # share of today's variance that comes from jumps
    df["jump_share"] = (df["rv5"] - df["bv5"]).clip(lower=0) / df["rv5"]
    df.loc[df["y"].isna(), "jump_share"] = np.nan                 # blank it on the bad days too

    # per-stock features: always inside groupby("symbol") so stocks never mix
    g = df.groupby("symbol")["y"]
    df["x_day"] = df["y"]
    df["x_week"] = g.transform(lambda s: s.rolling(5).mean())
    df["x_month"] = g.transform(lambda s: s.rolling(22).mean())
    df["target"] = g.shift(-1)                                    # tomorrow's log volatility

    all_features = sorted({f for fs in FEATURE_SETS.values() for f in fs})
    return df.dropna(subset=all_features + ["target"])            # dropna LAST


# ---------------------------------------------------------------- 2. models
def make_mlp(n_in, hidden):
    """hidden = list of layer sizes, e.g. [32, 16]. No activation on the output layer."""
    layers, size = [], n_in
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def to_tensor(a):
    return torch.tensor(np.array(a, dtype=np.float32))


def train_mlp(X_fit, y_fit, X_val, y_val, hidden, seed, lr, weight_decay, max_epochs, patience):
    """Train with early stopping: keep the weights from the epoch with the best VALIDATION loss."""
    torch.manual_seed(seed)
    model = make_mlp(X_fit.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        loss = loss_fn(model(X_fit), y_fit)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val), y_val).item()
        if val < best_val - 1e-6:
            best_val, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1


# ---------------------------------------------------------------- 3. main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--out", default="results_stocks.csv")
    p.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    seeds = [0, 1, 2, 3, 4]
    archs = [[8], [16, 16], [32, 32]]
    if args.quick:
        seeds, archs, args.max_epochs, args.patience = [0, 1], [[8]], 30, 10

    t0 = time.time()
    df = load_data(args.data)
    print(f"data ready: {len(df)} rows, {df['symbol'].nunique()} stocks  ({time.time() - t0:.0f}s)")

    rows = []                                                     # (cutoff, features, model, mse)
    for cutoff in CUTOFFS:
        train = df[df["date"] < cutoff]
        test = df[df["date"] >= cutoff]

        # validation = the last year of the training period (used ONLY to decide when to stop)
        val_start = pd.Timestamp(cutoff) - pd.DateOffset(years=1)
        fit = train[train["date"] < val_start]
        val = train[train["date"] >= val_start]
        print(f"\n{cutoff}: fit {len(fit)} | val {len(val)} | test {len(test)}")

        rows.append((cutoff, "-", "tomorrow = today", mean_squared_error(test["target"], test["x_day"])))

        for name, feats in FEATURE_SETS.items():
            # --- HAR (linear regression on the full training period)
            har = LinearRegression().fit(train[feats], train["target"])
            mse_har = mean_squared_error(test["target"], har.predict(test[feats]))
            rows.append((cutoff, name, "HAR", mse_har))
            print(f"  [{name}] HAR {mse_har:.5f}")

            # --- standardize with TRAINING-FIT statistics only (never the test data)
            mu, sd = fit[feats].mean(), fit[feats].std()
            X_fit, X_val, X_test = [to_tensor((d[feats] - mu) / sd) for d in (fit, val, test)]
            # standardize the target too (mean/std of the fit period), so the network does not
            # have to spend hundreds of steps just to reach the level of log-volatility (about -9)
            y_mu, y_sd = fit["target"].mean(), fit["target"].std()
            y_fit, y_val = [to_tensor((d["target"] - y_mu) / y_sd).reshape(-1, 1) for d in (fit, val)]
            y_test = test["target"].to_numpy()

            for hidden in archs:
                singles, preds = [], []
                for seed in seeds:
                    model, n_ep = train_mlp(X_fit, y_fit, X_val, y_val, hidden, seed,
                                            args.lr, args.weight_decay, args.max_epochs, args.patience)
                    with torch.no_grad():
                        pred = model(X_test).numpy().ravel() * y_sd + y_mu   # back to the original scale
                    preds.append(pred)
                    singles.append(mean_squared_error(y_test, pred))
                    print(f"    MLP {hidden} seed {seed}: {singles[-1]:.5f}  ({n_ep} epochs)")
                rows.append((cutoff, name, f"MLP {hidden} (mean of seeds)", float(np.mean(singles))))
                rows.append((cutoff, name, f"MLP {hidden} (ensemble)",
                             mean_squared_error(y_test, np.mean(preds, axis=0))))

    res = pd.DataFrame(rows, columns=["cutoff", "features", "model", "mse"])
    res.to_csv(args.out, index=False)
    table = res.pivot_table(index=["features", "model"], columns="cutoff", values="mse", sort=False)
    print("\n", table.round(5).to_string())
    print(f"\nsaved to {args.out}   total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
