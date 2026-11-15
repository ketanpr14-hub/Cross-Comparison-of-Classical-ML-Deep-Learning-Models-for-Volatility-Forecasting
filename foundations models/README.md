# Foundation models on VOLARE stocks

This script predicts the next session's `log(rv5)` for the 40 stocks. It uses
Quentin's stock-day exclusions, test cutoffs and HAR features. All predictions
are on the log-variance scale.

Use Python 3.13 and install the dependencies in a virtual environment. Run the
commands below from this folder (`cd "foundations models"` from the repo root).

```sh
python -m pip install "torch==2.11.0" numpy pandas scikit-learn \
  "chronos-forecasting==2.3.2" "timesfm[torch]==3.0.2" "granite-tsfm==0.3.12" \
  "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff" \
  "transformers==5.22.2" "peft==0.21.2"
```

On Apple Silicon, install `mlx==0.32.4` for TimesFM or pass `--device cpu`.
The model weights download on first use. Put the VOLARE CSV under
`data/realized_variance_stocks_20261030/` or give its path with `--data`.
If the CSV is in the folder next to the repo, pass
`--data ../../realized_variance_stocks_20261030/realized_variance_stocks_20261030.csv`.

```sh
# Check the setup on two stocks.
python foundation/stocks_foundation_models.py --quick

# Run all four models at both depths on the same stock-dates.
python foundation/stocks_foundation_models.py

# Add one optimized result per model and cutoff to the same CSV.
python foundation/stocks_foundation_models.py --optimize

# Compare frozen and fine-tuned weights without HAR.
python foundation/stocks_foundation_models.py --compare-adaptation

# Compare both depths on the original complete-window sample.
python foundation/stocks_foundation_models.py --strict-context

# Adapt Chronos and evaluate it on the 2021 cutoff.
python foundation/stocks_foundation_models.py --model chronos --context 1024 --finetune --out /tmp/stocks_lora.csv
```

## Sample and scoring

`load_panel` reads the stock CSV and applies Quentin's six exclusions.
Both depths use the same stock-dates, including the first 680 forecasts of
2019. The long run uses up to 1,024 past sessions. When fewer are available,
Chronos, Timestral and TimesFM receive the shorter history. TTM keeps its
fixed-size checkpoint and masks the leading padding. The `sample` column
labels the full sample and the complete-window comparison (`strict_1024`).

The CSV reports `n_forecasts`, MSE on `log(rv5)`, and QLIKE on the variance
scale. For each model and cutoff, `sigma2` is the sample variance of log
forecast errors in the preceding validation year (one session in five).
QLIKE uses the variance forecast `exp(predicted_log_rv5 + sigma2 / 2)`.
For the optimized variants, the second half of the validation year supplies
the residuals, after fitting any calibration on its first half.

## Adaptation

`--optimize` picks the depth and any calibration by validation MSE. It tries
22, 128, 256, 512 and 1,024 sessions, or the native TTM checkpoints at 52,
90, 512 and 1,024. The 2019 search stops at 512 because there are only 755
sessions per stock before 2018. Candidates include the raw forecast, linear
calibration and mixes with Quentin's market-and-jump HAR. The script also
tries LoRA for Chronos and head fine-tuning for Timestral and TTM at the
selected depth. TimesFM keeps its pretrained weights in this mode.

`--compare-adaptation` scores each model before and after fine-tuning without
HAR, at the same depth and on the same dates. The depths come from the
validation search for each cutoff. For 2019, they are 256 for Chronos and
Timestral, 22 for TimesFM and 52 for TTM. For 2021, they are 1,024 for Chronos,
128 for Timestral, 512 for TimesFM and 52 for TTM. Both modes use 2018 for
validation before the 2019 cutoff and 2020 before the 2021 cutoff. Weight
updates use only targets before the validation year. TimesFM head training
uses MLX on Apple Silicon.

Each run updates the relevant rows in `results/stocks_foundation_models.csv`.
Use `--cutoff` to run one cutoff or `--out` to choose another CSV. Without
`--out`, `--quick` writes to a temporary location.

For the baseline table, the earlier forecasts were matched by stock and
date, then rescored alongside the 680 new forecasts. Inference times appear
only where they cover the sample shown. The other experiments were rerun.
The Timestral adaptation runs used Apple GPU acceleration with normalization
on CPU. The script uses Timestral's CPU path on Mac, so its times can differ.
