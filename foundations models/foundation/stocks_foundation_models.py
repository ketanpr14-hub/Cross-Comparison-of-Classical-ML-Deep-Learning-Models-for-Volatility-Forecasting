import argparse
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import LinearRegression

FOUNDATION_DIR = Path(__file__).resolve().parent.parent
DATA = FOUNDATION_DIR / "data/realized_variance_stocks_20261030/realized_variance_stocks_20261030.csv"
RESULTS = FOUNDATION_DIR / "results/stocks_foundation_models.csv"
CUTOFFS = ("2019-01-01", "2021-01-01")
BAD_DAYS = [
    ("2023-01-24", "MCD"), ("2023-01-24", "NKE"), ("2023-01-24", "V"),
    ("2023-01-24", "WMT"), ("2023-01-24", "XOM"), ("2016-09-27", "PG"),
]
HAR_FEATURES = ["log_rv", "week", "month"]
HAR_MARKET_FEATURES = HAR_FEATURES + ["market_day", "market_week", "jump_share"]
HAR_WEIGHTS = {"har_25": 0.25, "har_50": 0.50, "har_75": 0.75}
CHECKPOINTS = {
    "chronos": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "timestral": ("mistralai/Timestral-209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"),
    "timesfm": ("google/timesfm-3.0-pytorch", "43046b85ec22d584a13f8098c2ed39c889e129c2"),
    "ttm": ("ibm-granite/granite-timeseries-ttm-r3", "7d99b7e35b3dc18f9db3b2b73cc9699dea6c9508"),
}
TTM_REVISIONS = {
    52: CHECKPOINTS["ttm"][1],
    90: "417bfc83545095031c2a002ddfe73ac6b700ea1a",
    512: "7b070728ff280ee2bb682bd8dfbe71da21ca2c7c",
    1024: "c6da085bc334cb37b268c2312bc45f859cd316c9",
}
ADAPTATION_DEPTHS = {"chronos": 1024, "timestral": 128, "timesfm": 512, "ttm": 52}
SCORE_COLUMNS = [
    "cutoff", "context", "model", "n_forecasts", "mse_log_rv5", "forecast_seconds",
    "adaptation", "validation_mse",
]


def load_stocks(path):
    stocks = pd.read_csv(path, usecols=["date", "symbol", "rv5", "bv5"], parse_dates=["date"])
    stocks = stocks.sort_values(["symbol", "date"]).reset_index(drop=True)
    if stocks.duplicated(["symbol", "date"]).any():
        raise ValueError("The stocks file contains duplicate symbol/date rows.")
    for date, symbol in BAD_DAYS:
        stocks.loc[(stocks["date"] == date) & (stocks["symbol"] == symbol), "rv5"] = np.nan

    stocks["log_rv"] = np.log(stocks["rv5"].where(stocks["rv5"] > 0))
    by_stock = stocks.groupby("symbol")["log_rv"]
    stocks["week"] = by_stock.transform(lambda series: series.rolling(5).mean())
    stocks["month"] = by_stock.transform(lambda series: series.rolling(22).mean())
    stocks["target"] = by_stock.shift(-1)
    stocks["target_date"] = stocks.groupby("symbol")["date"].shift(-1)

    market = stocks.groupby("date")["log_rv"].mean()
    stocks["market_day"] = stocks["date"].map(market)
    stocks["market_week"] = stocks["date"].map(market.rolling(5).mean())
    stocks["jump_share"] = (stocks["rv5"] - stocks["bv5"]).clip(lower=0) / stocks["rv5"]
    stocks["eligible"] = stocks[HAR_FEATURES + ["target", "market_week", "jump_share"]].notna().all(axis=1)
    return stocks


def make_windows(stocks, depth, rows):
    dates, contexts = [], []
    for _, stock in stocks.groupby("symbol", sort=False):
        if len(stock) < depth:
            continue
        # Fill gaps inside the history using only earlier sessions of this stock.
        history = stock["log_rv"].ffill().to_numpy()
        windows = np.lib.stride_tricks.sliding_window_view(history, depth)
        origins = stock.iloc[depth - 1:]
        keep = rows.loc[origins.index].to_numpy() & np.isfinite(windows).all(axis=1)
        dates.append(origins.loc[keep])
        contexts.append(windows[keep].astype(np.float32))
    if not dates or not any(len(part) for part in dates):
        raise ValueError(f"No eligible rows with {depth} sessions of history.")
    return pd.concat(dates, ignore_index=True), np.concatenate(contexts)


def model_device(name, device):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "mps" and name in ("timestral", "ttm"):
        return "cpu"
    return device


def load_model(name, depth, device, compile_mlx=True):
    repo, revision = CHECKPOINTS[name]
    if name == "ttm":
        revision = TTM_REVISIONS[max(52, depth)]
    path = snapshot_download(repo, revision=revision, token=False,
                             allow_patterns=["config.json", "model.safetensors"])

    if name == "chronos":
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(path, device_map=device, dtype=torch.float32)
    if name == "timestral":
        from timestral import TimestralForecaster

        return TimestralForecaster.from_pretrained(path, device=device)
    if name == "timesfm":
        if device == "mps":
            from timesfm3.mlx import TimesFM3Forecaster

            return TimesFM3Forecaster.from_pretrained(path, compile=compile_mlx)
        from timesfm3 import TimesFM3Forecaster

        return TimesFM3Forecaster.from_pretrained(path, device=device)
    from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import (
        TinyTimeMixerForDecomposedPrediction,
        TinyTimeMixerForPrediction,
    )

    model_type = TinyTimeMixerForPrediction if depth == 1024 else TinyTimeMixerForDecomposedPrediction
    return model_type.from_pretrained(path).to(device).eval()


def predict_batch(name, model, batch, device):
    if name == "chronos":
        quantiles, _ = model.predict_quantiles(
            batch[:, None, :], prediction_length=1, quantile_levels=[0.5],
            batch_size=len(batch), context_length=batch.shape[1], cross_learning=False,
        )
        return torch.stack(quantiles).cpu().numpy().reshape(-1)
    if name == "timestral":
        return model.forecast(batch, prediction_length=1).median.cpu().numpy().reshape(-1)
    if name == "timesfm":
        # Two batched calls give the symmetric forecast without looping over series.
        positive = model.predict_batch(list(batch), horizon=1)
        negative = model.predict_batch(list(-batch), horizon=1)
        return np.array([(p.forecast.item() - n.forecast.item()) / 2
                         for p, n in zip(positive, negative, strict=True)])
    return ttm_median(model, batch, device).cpu().numpy()


def ttm_median(model, batch, device):
    values = torch.from_numpy(batch).to(device).unsqueeze(-1)
    output = model(past_values=values, past_observed_mask=torch.ones_like(values, dtype=torch.bool),
                   return_loss=False)
    median = list(model.config.quantile_levels).index(0.5)
    return output.quantile_outputs[:, median, 0, 0]


@torch.inference_mode()
def forecast(name, model, contexts, device, batch_size):
    predictions = []
    for start in range(0, len(contexts), batch_size):
        batch = np.ascontiguousarray(contexts[start:start + batch_size])
        predictions.append(predict_batch(name, model, batch, device))
        if (start // batch_size + 1) % 50 == 0:
            print(f"  {min(start + batch_size, len(contexts)):,}/{len(contexts):,} forecasts", flush=True)
    predictions = np.concatenate(predictions)
    if predictions.shape != (len(contexts),) or not np.isfinite(predictions).all():
        raise ValueError(f"{name} returned invalid predictions.")
    return predictions


# 3. Optional Chronos adaptation
def chronos_training_series(stocks, before, depth, quick):
    train = stocks.loc[stocks["date"] < before]
    series = [stock["log_rv"].to_numpy(dtype=np.float32)
              for _, stock in train.groupby("symbol") if len(stock) > depth]
    if not series:
        raise ValueError(f"No training series has more than {depth} sessions before {before}.")
    return series[:2] if quick else series


def train_chronos(model, series, depth, steps, quick, validation=None):
    import peft

    print(f"  LoRA: {len(series)} training series, {steps} updates", flush=True)
    with TemporaryDirectory(prefix="chronos-lora-") as directory:
        return model.fit(
            inputs=series, validation_inputs=validation,
            context_length=depth, min_past=depth, prediction_length=1,
            finetune_mode="lora", learning_rate=1e-5, num_steps=steps,
            batch_size=2 if quick else 8, seed=0, optim="adamw_torch", dataloader_pin_memory=False,
            eval_steps=min(25, steps), save_steps=min(25, steps), output_dir=directory,
            disable_tqdm=True,
        )


def finetune_chronos(model, stocks, cutoff, depth, steps, quick):
    val_start = pd.Timestamp(cutoff) - pd.DateOffset(years=1)
    series = chronos_training_series(stocks, val_start, depth, quick)

    in_validation = stocks["target_date"].ge(val_start) & stocks["target_date"].lt(cutoff)
    dates = stocks.loc[in_validation, "target_date"].sort_values().drop_duplicates()
    rows = stocks["eligible"] & stocks["target_date"].isin(dates.iloc[::20])
    if quick:
        rows &= stocks["symbol"].isin(stocks["symbol"].unique()[:2])
    validation, history = make_windows(stocks, depth, rows)
    inputs = np.column_stack([history, validation["target"].to_numpy(dtype=np.float32)])
    return train_chronos(model, series, depth, steps, quick, inputs[:, None, :])


def timestral_training_median(model, batch, device):
    from timestral.forecaster import pad_to_patch

    values = torch.from_numpy(batch).to(device)
    observed = torch.ones_like(values, dtype=torch.bool)
    values, observed = pad_to_patch(values, observed, model.patch_size)
    groups = torch.arange(len(batch), device=device)
    positive = model.model.predict_block(values, observed, groups, horizon_patches=1)
    negative = model.model.predict_block(-values, observed, groups, horizon_patches=1)
    return (positive[:, 0, model.median_index] - negative[:, 0, model.median_index]) / 2


def head_training_windows(stocks, depth):
    train_rows = stocks["eligible"] & stocks["target_date"].lt("2020-01-01")
    train, contexts = make_windows(stocks, depth, train_rows)
    return contexts, train["target"].to_numpy(dtype=np.float32)


def finetune_head(name, model, stocks, depth, device, steps):
    contexts, targets = head_training_windows(stocks, depth)

    if name == "timestral":
        network = model.model
        heads = [network.head]
        predict, learning_rate = timestral_training_median, 1e-5
    else:
        network = model
        heads = [model.head] if depth == 1024 else [model.trend_forecaster.head, model.residual_forecaster.head]
        predict, learning_rate = ttm_median, 1e-4
    network.requires_grad_(False)
    for head in heads:
        head.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=learning_rate)
    rng = np.random.default_rng(0)
    print(f"  {name} head: {steps} updates", flush=True)
    for _ in range(steps):
        batch = rng.integers(len(targets), size=16)
        actual = torch.from_numpy(targets[batch]).to(device)
        predicted = predict(model, contexts[batch], device)
        loss = (predicted - actual).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def finetune_timesfm(stocks, depth, steps):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    contexts, targets = head_training_windows(stocks, depth)
    forecaster = load_model("timesfm", depth, "mps", compile_mlx=False)
    network = forecaster.model
    network.freeze()
    network.output_head.unfreeze()
    median = forecaster.config.median_quantile_index

    def loss(context, actual):
        positive = mx.sort(network.decode(context, horizon=1), axis=-1)[:, 0, 0, median]
        negative = mx.sort(network.decode(-context, horizon=1), axis=-1)[:, 0, 0, median]
        return mx.mean(mx.square((positive - negative) / 2 - actual))

    optimizer = optim.AdamW(learning_rate=1e-5)
    loss_and_grad = nn.value_and_grad(network, loss)
    rng = np.random.default_rng(0)
    print(f"  timesfm head: {steps} updates", flush=True)
    for _ in range(steps):
        batch = rng.integers(len(targets), size=16)
        error, gradients = loss_and_grad(mx.array(contexts[batch, None, :]), mx.array(targets[batch]))
        optimizer.update(network, gradients)
        mx.eval(error, network.trainable_parameters(), optimizer.state)
        if not np.isfinite(error.item()):
            raise ValueError("TimesFM fine-tuning loss is not finite.")

    tuned = load_model("timesfm", depth, "mps")
    tuned.model.output_head.update(network.output_head.parameters())
    return tuned


def adapt_model(name, model, stocks, depth, device, args):
    steps = 1 if args.quick else args.steps
    if name == "chronos":
        series = chronos_training_series(stocks, "2020-01-01", depth, args.quick)
        return train_chronos(model, series, depth, steps, args.quick), "lora"
    if name == "timesfm":
        return finetune_timesfm(stocks, depth, steps), "head_tuned"
    tuned = load_model(name, depth, device)
    return finetune_head(name, tuned, stocks, depth, device, steps), "head_tuned"


def calibration_inputs(data, predictions, method):
    if method == "affine":
        return predictions[:, None]
    return np.column_stack([predictions, data[HAR_FEATURES].to_numpy()])


def choose_calibration(validation, predictions, har_predictions):
    fit = validation["target_date"].lt("2020-07-01").to_numpy()
    actual = validation["target"].to_numpy()
    best = (np.mean((actual[~fit] - predictions[~fit]) ** 2), "raw", None)
    for method in ("affine", "with_har"):
        features = calibration_inputs(validation, predictions, method)
        regression = LinearRegression().fit(features[fit], actual[fit])
        mse = np.mean((actual[~fit] - regression.predict(features[~fit])) ** 2)
        if mse < best[0]:
            best = (mse, method, regression)
    for method, weight in HAR_WEIGHTS.items():
        blended = (1 - weight) * predictions[~fit] + weight * har_predictions[~fit]
        mse = np.mean((actual[~fit] - blended) ** 2)
        if mse < best[0]:
            best = (mse, method, None)
    return best


def optimized_scores(stocks, names, args):
    cutoff = "2021-01-01"
    in_validation = stocks["target_date"].ge("2020-01-01") & stocks["target_date"].lt(cutoff)
    val_dates = stocks.loc[in_validation, "target_date"].sort_values().drop_duplicates()
    stride = 40 if args.quick else 5
    val_rows = stocks["eligible"] & stocks["target_date"].isin(val_dates.iloc[::stride])
    test_rows = stocks["eligible"] & stocks["date"].ge(cutoff)
    if args.quick:
        two_stocks = stocks["symbol"].isin(stocks["symbol"].unique()[:2])
        val_rows &= two_stocks
        first_test_rows = stocks.loc[test_rows & two_stocks].groupby("symbol").head(2)
        test_rows &= stocks.index.isin(first_test_rows.index)
    validation, val_context = make_windows(stocks, 1024, val_rows)
    test, test_context = make_windows(stocks, 1024, test_rows)
    train = stocks.loc[stocks["eligible"] & stocks["target_date"].lt("2020-01-01")]
    har = LinearRegression().fit(train[HAR_MARKET_FEATURES], train["target"])
    val_har = har.predict(validation[HAR_MARKET_FEATURES])
    scores = []
    for name in names:
        device = model_device(name, args.device)
        depths = (52, 90, 512, 1024) if name == "ttm" else (22, 128, 256, 512, 1024)
        best_mse = np.inf
        for depth in depths:
            model = load_model(name, depth, device)
            predictions = forecast(name, model, val_context[:, -depth:], device, args.batch_size)
            mse, method, regression = choose_calibration(validation, predictions, val_har)
            print(f"{name} {depth} sessions, {method}: validation MSE {mse:.5f}", flush=True)
            if mse < best_mse:
                best_mse, best_depth, best_model = mse, depth, model
                best_method, best_regression = method, regression
        del model

        adaptation = "frozen"
        if name in ("chronos", "ttm", "timestral"):
            tuned, tuned_method = adapt_model(name, best_model, stocks, best_depth, device, args)
            predictions = forecast(name, tuned, val_context[:, -best_depth:], device, args.batch_size)
            mse, method, regression = choose_calibration(validation, predictions, val_har)
            print(f"{name} {best_depth} sessions, fine-tuned + {method}: validation MSE {mse:.5f}", flush=True)
            if mse < best_mse:
                best_mse, best_model = mse, tuned
                best_method, best_regression = method, regression
                adaptation = tuned_method
            del tuned

        print(f"{name}: {best_depth} sessions, {adaptation} + {best_method}, {len(test):,} test forecasts", flush=True)
        started = perf_counter()
        predictions = forecast(name, best_model, test_context[:, -best_depth:], device, args.batch_size)
        if best_regression is not None:
            predictions = best_regression.predict(calibration_inputs(test, predictions, best_method))
        if best_method in HAR_WEIGHTS:
            weight = HAR_WEIGHTS[best_method]
            predictions = (1 - weight) * predictions + weight * har.predict(test[HAR_MARKET_FEATURES])
        add_scores(scores, test, predictions, (cutoff,), best_depth, f"{name}_optimized",
                   perf_counter() - started, f"{adaptation}+{best_method}", best_mse)
        del best_model
    return scores


def compare_adaptation(stocks, names, args):
    cutoff = "2021-01-01"
    rows = stocks["eligible"] & stocks["date"].ge(cutoff)
    if args.quick:
        sample = stocks.loc[rows].groupby("symbol").head(2).head(4)
        rows &= stocks.index.isin(sample.index)
    test, contexts = make_windows(stocks, max(ADAPTATION_DEPTHS.values()), rows)
    scores = []
    for name in names:
        depth = ADAPTATION_DEPTHS[name]
        device = model_device(name, args.device)
        model = load_model(name, depth, device)
        print(f"{name}: {depth} sessions, frozen, {len(test):,} forecasts", flush=True)
        started = perf_counter()
        frozen = forecast(name, model, contexts[:, -depth:], device, args.batch_size)
        add_scores(scores, test, frozen, (cutoff,), depth, f"{name}_solo_frozen",
                   perf_counter() - started, "frozen_no_har")

        tuned, method = adapt_model(name, model, stocks, depth, device, args)
        del model
        print(f"{name}: {depth} sessions, {method}, {len(test):,} forecasts", flush=True)
        started = perf_counter()
        adapted = forecast(name, tuned, contexts[:, -depth:], device, args.batch_size)
        add_scores(scores, test, adapted, (cutoff,), depth, f"{name}_solo_{method}",
                   perf_counter() - started, f"{method}_no_har")
        del tuned
    return scores


def add_scores(scores, test, predictions, cutoffs, depth, name, forecast_seconds=None,
               adaptation=None, validation_mse=None):
    for cutoff in cutoffs:
        keep = test["date"].ge(cutoff).to_numpy()
        actual = test.loc[keep, "target"].to_numpy()
        mse = np.mean((actual - predictions[keep]) ** 2)
        scores.append((cutoff, depth, name, len(actual), mse, forecast_seconds, adaptation, validation_mse))


def save_scores(scores, out, append=False):
    table = pd.DataFrame(scores, columns=SCORE_COLUMNS)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    if append and out.is_file():
        previous = pd.read_csv(out, dtype=str)
        previous = previous.loc[~previous["model"].isin(table["model"])]
        table = pd.concat([previous, table], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as directory:
        temporary_csv = Path(directory) / out.name
        table.to_csv(temporary_csv, index=False)
        temporary_csv.replace(out)
    print(f"Results in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, help="output CSV path")
    parser.add_argument("--model", choices=["all", *CHECKPOINTS], default="all")
    parser.add_argument("--context", type=int, nargs="+", choices=[22, 1024], default=[22, 1024],
                        help="context lengths for the baseline or standalone LoRA run")
    parser.add_argument("--cutoff", choices=CUTOFFS, help="one test cutoff instead of both")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="two stocks and a few test dates")
    parser.add_argument("--finetune", action="store_true", help="adapt Chronos-2 with LoRA before scoring")
    parser.add_argument("--optimize", action="store_true", help="add one 2021 result per model after tuning on 2020")
    parser.add_argument("--compare-adaptation", action="store_true", help="compare frozen and adapted models without HAR")
    parser.add_argument("--steps", type=int, default=100, help="number of fine-tuning updates")
    args = parser.parse_args()

    depths = sorted(set(args.context))
    if args.finetune and (args.model != "chronos" or len(depths) != 1):
        parser.error("--finetune requires --model chronos and one context length")
    if sum((args.optimize, args.finetune, args.compare_adaptation)) > 1:
        parser.error("Choose one of --optimize, --finetune and --compare-adaptation.")
    if (args.optimize or args.compare_adaptation) and args.cutoff == "2019-01-01":
        parser.error("This experiment uses the 2021 test cutoff.")
    if min(args.batch_size, args.threads, args.steps) < 1:
        parser.error("--batch-size, --threads and --steps must be positive")
    cutoff = args.cutoff or ("2021-01-01" if args.finetune else None)
    cutoffs = (cutoff,) if cutoff else CUTOFFS
    if not args.data.is_file():
        parser.error(f"CSV not found at {args.data}. Pass its path with --data.")

    torch.set_num_threads(args.threads)
    stocks = load_stocks(args.data)
    out = args.out or RESULTS
    if args.quick and args.out is None:
        out = Path(gettempdir()) / "stocks_foundation_models_quick.csv"
    names = list(CHECKPOINTS) if args.model == "all" else [args.model]
    if args.optimize:
        save_scores(optimized_scores(stocks, names, args), out, append=True)
        return
    if args.compare_adaptation:
        if "timesfm" in names and model_device("timesfm", args.device) != "mps":
            parser.error("TimesFM head training uses MLX on Apple Silicon. Run with --device auto on that Mac.")
        save_scores(compare_adaptation(stocks, names, args), out, append=True)
        return

    test, contexts = make_windows(stocks, max(depths), stocks["eligible"] & stocks["date"].ge(cutoffs[0]))
    if args.quick:
        small = test[test["symbol"].isin(test["symbol"].unique()[:2])]
        positions = [small.loc[small["date"].ge(day)].groupby("symbol").head(4).index for day in cutoffs]
        keep = np.unique(np.concatenate(positions))
        test, contexts = test.iloc[keep].reset_index(drop=True), contexts[keep]
    if not test["date"].ge(cutoffs[-1]).any():
        parser.error("No eligible test rows at the requested cutoff.")

    scores = []
    add_scores(scores, test, test["log_rv"].to_numpy(), cutoffs, 1, "persistence")
    for day in cutoffs:
        train = stocks.loc[stocks["eligible"] & stocks["date"].lt(day)]
        # Match Quentin's original HAR split by forecast origin.
        har = LinearRegression().fit(train[HAR_FEATURES], train["target"])
        add_scores(scores, test, har.predict(test[HAR_FEATURES]), (day,), 22, "HAR")

    for depth in depths:
        for name in names:
            device = model_device(name, args.device)
            print(f"{name}: {depth} sessions, {len(test):,} forecasts, {device}", flush=True)
            model = load_model(name, depth, device)
            if args.finetune:
                steps = 1 if args.quick else args.steps
                model = finetune_chronos(model, stocks, cutoff, depth, steps, args.quick)
            started = perf_counter()
            predictions = forecast(name, model, contexts[:, -depth:], device, args.batch_size)
            forecast_seconds = perf_counter() - started
            model_name = f"{name}_lora" if args.finetune else name
            add_scores(scores, test, predictions, cutoffs, depth, model_name, forecast_seconds)
            print(f"  forecast time {forecast_seconds:.1f}s", flush=True)
            del model

    save_scores(scores, out)


if __name__ == "__main__":
    main()
