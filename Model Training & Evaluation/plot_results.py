"""
Plot forecast losses for every model and feature set, relative to the naive baseline.

Reads:  results/forecasts/*.parquet  (baseline_models.py, ridge_lasso.py, knn.py, and later
        the forest/boosting scripts; all must share the long format with 'model', and
        'feature_set' where the model uses one)
Writes: results/figures/<metric>_summary.html  median ratio to naive per asset class
        results/figures/<metric>_by_symbol.html  ratio per symbol, model and feature set
        results/tables/results.xlsx              both tables (QLIKE and MSE), one sheet each

Ratios are computed per symbol against naive on exactly the same dates, so values
below 1 mean the model beats naive. Run from the project root:  python plot_results.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from plotly.subplots import make_subplots
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path

from config.config import ASSET_CLASSES, FIG_DIR, RESULTS_DIR

TABLE_DIR = RESULTS_DIR.parent / "tables"  # results/tables/results.xlsx

METRIC = "qlike"  # or "mse"

# Display order; any model not listed (e.g. a new forest script) is added at the end
MODEL_ORDER = ["har", "ridge", "lasso", "knn_level", "knn_change", "rf", "gbm"]
MODEL_LABELS = {"har": "HAR (OLS)", "ridge": "Ridge", "lasso": "Lasso",
                "knn_level": "k-NN (level)", "knn_change": "k-NN (change)",
                "rf": "Random forest", "gbm": "Gradient boosting"}

# One fixed color + marker per feature set, so every chart encodes them the same way
SET_ORDER = ["har", "har_ext", "har_mkt"]
SET_STYLE = {
    "har":     {"label": "HAR terms",            "color": "#2a78d6", "symbol": "circle"},
    "har_ext": {"label": "+ extras",             "color": "#eb6834", "symbol": "diamond"},
    "har_mkt": {"label": "+ extras + market",    "color": "#1baf7a", "symbol": "square"},
}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
# Diverging scale for ratios: blue = better than naive, gray = same, red = worse
RATIO_SCALE = [[0.0, "#5598e7"], [0.5, "#f0efec"], [1.0, "#e66767"]]


# ---------- Losses ----------

def loss(actual, forecast, metric=METRIC):
    if metric == "mse":
        return np.mean((actual - forecast) ** 2)
    ratio = np.exp(2 * actual) / np.exp(2 * forecast)  # QLIKE on variances; inputs are log vols
    return np.mean(ratio - np.log(ratio) - 1)


# ---------- Data ----------

def load_forecasts() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No forecast files in {RESULTS_DIR}")
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

    if "feature_set" not in f.columns:
        f["feature_set"] = np.nan
    # Baseline HAR uses the HAR terms; naive uses no features
    f.loc[f["model"].eq("har") & f["feature_set"].isna(), "feature_set"] = "har"
    f["feature_set"] = f["feature_set"].fillna("none")

    keys = ["asset_class", "symbol", "model", "feature_set", "date"]
    dupes = f.duplicated(keys).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate forecasts; check for old result files in {RESULTS_DIR}")
    return f


def summarize(f: pd.DataFrame, metric=METRIC) -> pd.DataFrame:
    """Loss per symbol, model and feature set, plus its ratio to naive on the same dates."""
    naive = f[f["model"] == "naive"][["asset_class", "symbol", "date", "forecast"]]
    if naive.empty:
        raise ValueError("No naive forecasts found; run baseline_models.py first")
    naive = naive.rename(columns={"forecast": "naive_forecast"})

    m = f[f["model"] != "naive"].merge(naive, on=["asset_class", "symbol", "date"], how="left")
    missing = m["naive_forecast"].isna().sum()
    if missing:
        print(f"warning: {missing} forecasts have no naive forecast on the same date; dropped")
        m = m.dropna(subset=["naive_forecast"])

    out = (
        m.groupby(["asset_class", "symbol", "model", "feature_set"])
        .apply(lambda g: pd.Series({
            "loss": loss(g["actual"], g["forecast"], metric),
            "naive_loss": loss(g["actual"], g["naive_forecast"], metric),
            "n": len(g),
        }), include_groups=False)
        .reset_index()
    )
    out["ratio"] = out["loss"] / out["naive_loss"]
    return out


def model_order(models) -> list[str]:
    present = list(dict.fromkeys(models))
    return [m for m in MODEL_ORDER if m in present] + [m for m in present if m not in MODEL_ORDER]


def classes_in(summary) -> list[str]:
    return [c for c in ASSET_CLASSES if c in summary["asset_class"].unique()]


# ---------- Styling ----------

def style(fig: go.Figure, title: str, subtitle: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:12px;color:{INK_MUTED}'>{subtitle}</span>",
                   font=dict(size=16, color=INK), x=0, xanchor="left"),
        template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height,
        font=dict(family="Inter, Arial, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=130, r=30, t=130, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    for ann in fig.layout.annotations:  # subplot titles
        if ann.text in [c.upper() for c in ASSET_CLASSES]:
            ann.font = dict(size=13, color=INK)
    return fig


# ---------- Figure 1: median ratio per asset class ----------

def plot_summary(summary: pd.DataFrame, metric=METRIC) -> go.Figure:
    """One panel per asset class. Each row is a model; one marker per feature set at the
    median ratio across symbols, with a line spanning the best to worst symbol."""
    classes = classes_in(summary)
    models = model_order(summary["model"])
    sets = [s for s in SET_ORDER if s in summary["feature_set"].unique()]
    offset = {s: (i - (len(sets) - 1) / 2) * 0.22 for i, s in enumerate(sets)}

    agg = (summary.groupby(["asset_class", "model", "feature_set"])["ratio"]
           .agg(median="median", low="min", high="max", n_symbols="count").reset_index())

    fig = make_subplots(rows=1, cols=len(classes), shared_yaxes=True,
                        subplot_titles=[c.upper() for c in classes], horizontal_spacing=0.04)
    for col, asset_class in enumerate(classes, start=1):
        fig.add_vline(x=1, line_color=INK_MUTED, line_width=1, line_dash="dash", row=1, col=col)
        for s in sets:
            d = agg[(agg["asset_class"] == asset_class) & (agg["feature_set"] == s)]
            d = d[d["model"].isin(models)]
            y = [models.index(m) + offset[s] for m in d["model"]]
            st = SET_STYLE[s]
            fig.add_scatter(
                x=d["median"], y=y, mode="markers", row=1, col=col,
                name=st["label"], legendgroup=s, showlegend=(col == 1),
                marker=dict(size=10, color=st["color"], symbol=st["symbol"],
                            line=dict(color=SURFACE, width=2)),
                error_x=dict(type="data", symmetric=False, array=d["high"] - d["median"],
                             arrayminus=d["median"] - d["low"], color=st["color"],
                             thickness=1.5, width=0),
                customdata=np.stack([d["model"].map(lambda m: MODEL_LABELS.get(m, m)),
                                     d["low"], d["high"], d["n_symbols"]], axis=-1),
                hovertemplate=(f"<b>%{{customdata[0]}}</b> · {st['label']}<br>"
                               "Median ratio %{x:.3f}<br>"
                               "Range %{customdata[1]:.3f} to %{customdata[2]:.3f}<br>"
                               "%{customdata[3]} symbols<extra></extra>"),
            )
        fig.update_xaxes(title_text=f"{metric.upper()} ratio to naive", row=1, col=col,
                         showgrid=True, gridcolor=GRID, zeroline=False,
                         tickfont=dict(color=INK_MUTED))

    fig.update_yaxes(tickvals=list(range(len(models))),
                     ticktext=[MODEL_LABELS.get(m, m) for m in models],
                     autorange="reversed", showgrid=False, tickfont=dict(color=INK))
    fig.update_yaxes(range=[len(models) - 0.5, -0.5])
    fig.update_xaxes(matches="x")  # same ratio scale in every panel, so classes compare directly
    return style(fig, f"{metric.upper()} relative to naive, by asset class",
                 "Marker = median across symbols · line = best to worst symbol · below 1 beats naive",
                 height=170 + 62 * len(models))


# ---------- Figure 2: ratio per symbol ----------

def plot_by_symbol(summary: pd.DataFrame, metric=METRIC) -> go.Figure:
    """Heatmap: rows = model · feature set, columns = symbols, one panel per asset class."""
    classes = classes_in(summary)
    models = model_order(summary["model"])
    summary = summary.copy()
    summary["row"] = summary["model"].map(lambda m: MODEL_LABELS.get(m, m)) + " · " + \
        summary["feature_set"].map(lambda s: SET_STYLE.get(s, {"label": s})["label"])
    rows = (summary.assign(mi=summary["model"].map(models.index),
                           si=summary["feature_set"].map(lambda s: SET_ORDER.index(s)
                                                         if s in SET_ORDER else 99))
            .sort_values(["mi", "si"])["row"].unique().tolist())

    spread = float(np.nanmax(np.abs(summary["ratio"] - 1))) or 0.1
    widths = [summary.loc[summary["asset_class"] == c, "symbol"].nunique() for c in classes]
    fig = make_subplots(rows=1, cols=len(classes), shared_yaxes=True,
                        column_widths=widths, horizontal_spacing=0.03,
                        subplot_titles=[c.upper() for c in classes])
    for col, asset_class in enumerate(classes, start=1):
        grid = (summary[summary["asset_class"] == asset_class]
                .pivot(index="row", columns="symbol", values="ratio")
                .reindex(rows))
        fig.add_heatmap(
            z=grid.values, x=grid.columns.tolist(), y=grid.index.tolist(),
            coloraxis="coloraxis", xgap=2, ygap=2,
            texttemplate="%{z:.2f}", textfont=dict(size=11, color=INK),
            hovertemplate="<b>%{x}</b><br>%{y}<br>Ratio to naive %{z:.3f}<extra></extra>",
            row=1, col=col,
        )
    fig.update_layout(coloraxis=dict(
        colorscale=RATIO_SCALE, cmin=1 - spread, cmax=1 + spread, cmid=1,
        colorbar=dict(title=dict(text="Ratio<br>to naive", font=dict(size=11)),
                      thickness=12, len=0.6, tickfont=dict(color=INK_MUTED)),
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK))
    fig.update_xaxes(tickfont=dict(color=INK_MUTED), showgrid=False)
    fig = style(fig, f"{metric.upper()} relative to naive, by symbol",
                "Blue = better than naive · gray = same · red = worse",
                height=170 + 30 * len(rows))
    fig.update_layout(showlegend=False, margin=dict(l=210, r=30, t=110, b=60))
    return fig


# ---------- Main ----------

def tables_for_excel(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """By-symbol and summary tables with both QLIKE and MSE, sorted for reading."""
    parts = []
    for metric in ["qlike", "mse"]:
        s = summarize(f, metric).rename(columns={
            "loss": metric, "naive_loss": f"naive_{metric}", "ratio": f"{metric}_ratio"})
        parts.append(s.set_index(["asset_class", "symbol", "model", "feature_set"]))
    by_symbol = parts[0].join(parts[1].drop(columns="n")).reset_index()
    by_symbol["n"] = by_symbol["n"].astype(int)

    # Readable labels and a sensible row order
    models = model_order(by_symbol["model"])
    by_symbol.insert(3, "model_label", by_symbol["model"].map(lambda m: MODEL_LABELS.get(m, m)))
    by_symbol.insert(5, "feature_set_label",
                     by_symbol["feature_set"].map(lambda s: SET_STYLE.get(s, {"label": s})["label"]))
    order = {
        "ac": by_symbol["asset_class"].map(lambda c: ASSET_CLASSES.index(c) if c in ASSET_CLASSES else 99),
        "mi": by_symbol["model"].map(models.index),
        "si": by_symbol["feature_set"].map(lambda s: SET_ORDER.index(s) if s in SET_ORDER else 99),
    }
    by_symbol = (by_symbol.assign(**order)
                 .sort_values(["ac", "mi", "si", "symbol"])
                 .drop(columns=list(order)).reset_index(drop=True))
    by_symbol = by_symbol[["asset_class", "symbol", "model", "model_label", "feature_set",
                           "feature_set_label", "n", "qlike", "naive_qlike", "qlike_ratio",
                           "mse", "naive_mse", "mse_ratio"]]

    summary = (
        by_symbol.groupby(["asset_class", "model", "model_label", "feature_set", "feature_set_label"],
                          sort=False)
        .agg(n_symbols=("symbol", "nunique"),
             median_qlike_ratio=("qlike_ratio", "median"),
             min_qlike_ratio=("qlike_ratio", "min"),
             max_qlike_ratio=("qlike_ratio", "max"),
             median_mse_ratio=("mse_ratio", "median"),
             min_mse_ratio=("mse_ratio", "min"),
             max_mse_ratio=("mse_ratio", "max"),
             beats_naive_qlike=("qlike_ratio", lambda r: int((r < 1).sum())))
        .reset_index()
    )
    return summary, by_symbol


def export_excel(f: pd.DataFrame) -> Path:
    """Write the summary and by-symbol tables to one workbook, one sheet each."""
    summary, by_symbol = tables_for_excel(f)
    path = TABLE_DIR / "results.xlsx"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in [("Summary", summary), ("By symbol", by_symbol)]:
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col_idx, col in enumerate(df.columns, start=1):
                letter = ws.cell(row=1, column=col_idx).column_letter
                width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) + 2
                ws.column_dimensions[letter].width = min(width, 32)
                if pd.api.types.is_float_dtype(df[col]):
                    for cell in ws[letter][1:]:
                        cell.number_format = "0.0000"
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    forecasts = load_forecasts()
    summary = summarize(forecasts)

    path = export_excel(forecasts)
    print(f"Tables saved to {path}")

    table = summary.pivot_table(index=["model", "feature_set"], columns="asset_class",
                                values="ratio", aggfunc="median")
    print(f"Median {METRIC.upper()} ratio to naive across symbols:")
    print(table.round(3))

    fig1, fig2 = plot_summary(summary), plot_by_symbol(summary)
    fig1.write_html(FIG_DIR / f"{METRIC}_summary.html")
    fig2.write_html(FIG_DIR / f"{METRIC}_by_symbol.html")
    fig1.show()
    fig2.show()


if __name__ == "__main__":
    main()