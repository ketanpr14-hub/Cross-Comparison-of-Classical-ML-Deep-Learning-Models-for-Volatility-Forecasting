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

# Add one optimized result per model to the same CSV.
python foundation/stocks_foundation_models.py --optimize

# Compare frozen and fine-tuned weights without HAR.
python foundation/stocks_foundation_models.py --compare-adaptation

# Reproduce the original short-context run.
python foundation/stocks_foundation_models.py --context 22 --out /tmp/stocks_22.csv

# Adapt Chronos and evaluate it on the 2021 cutoff.
python foundation/stocks_foundation_models.py --model chronos --context 1024 --finetune --out /tmp/stocks_lora.csv
```

With both context lengths, the script scores only dates with 1,024 sessions of
history. It fills missing historical values with the last available value for
that stock. TTM uses its 52-session checkpoint with internal padding in the short
run and its 1,024-session checkpoint in the long run. Timestral runs on CPU on
Apple Silicon, so the long run can take longer than the earlier GPU experiment.

Chronos LoRA fits on data before the last year preceding the cutoff. That last
year supplies validation targets, and test forecasts start at the cutoff.
Compare the LoRA scores with `--model chronos --context 1024 --cutoff 2021-01-01`.
Add `--quick` to check the training path with one update.

`--optimize` tests 22, 128, 256, 512 and 1,024 sessions. For TTM it uses the
native checkpoints at 52, 90, 512 and 1,024 sessions. It compares the raw forecast,
a linear calibration, a regression using the forecast and the three HAR
features, and blends with Quentin's market-and-jump HAR (25%, 50% or 75% HAR).
The blend uses a HAR fitted before 2020. Calibration uses January–June 2020.
July–December 2020 selects the depth and calibration. Both halves use one
session in five. At the selected depth, the script also tries Chronos LoRA and
forecast-head fine-tuning for TTM and Timestral, using data before 2020.
TimesFM 3.0 keeps its pretrained weights in this experiment.

The winning variant for each model is evaluated from the 2021 cutoff on the
same dates as the baseline table. `adaptation` records what was selected, and
`validation_mse` records its score on July–December 2020. `--optimize` preserves
existing baseline rows and replaces previous optimized rows for the models run.

`--compare-adaptation` appends a pair per model at the depth selected above
(Chronos 1,024, Timestral 128, TimesFM 512, TTM 52). Each pair uses the same
2021 test rows, input history and prediction code. Chronos uses LoRA, while
Timestral, TimesFM and TTM train only their forecast heads. All weight updates
use targets before 2020. TimesFM head training uses MLX on Apple Silicon.
The `_solo_` rows contain only the foundation model's forecast.

Each run writes one CSV to `results/stocks_foundation_models.csv`. It has one row
per model, depth and cutoff. `forecast_seconds` times inference on the full set
for each model and depth. The two cutoffs reuse those forecasts. `--out` takes a
CSV path. Without `--out`, `--quick` writes to the system's temporary directory
so it won't replace the full results. The baseline run starts a new table.
The two adaptation commands add their rows to that table.

The supplied CSV contains the results of the earlier full runs. The 22-session
times were measured on the original 78,605 forecasts, before aligning the scores
to the 77,925 dates common to both depths.
